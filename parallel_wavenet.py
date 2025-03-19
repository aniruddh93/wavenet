# Implementation of Parallel wavenet model
# Parallel wavenet uses a student model (Ps) and a trained teacher model (Pt)
# and minimizes the KL-div b/w them.
# P(x_t/x_0...x_t-1) is modelled as mixture of logistic distribution.

import numpy as np
from scipy.io import wavfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('/kaggle/input/wavenet_base_model/pytorch/default/1/')

from wavenet_model import WavenetConfig, WavenetModel


class ParallelWavenetDataset(Dataset):
    """Provides noise variates for training the parallel wavenet."""

    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.dist = torch.distributions.TransformedDistribution(
            torch.distributions.Uniform(0, 1),
            [torch.distributions.transforms.SigmoidTransform().inv,
             torch.distributions.transforms.AffineTransform(loc=0.0, scale=1.0)]
        )

    def __len__(self):
        return 100
    
    def __getitem__(self, index):
        return self.dist.sample((self.seq_len, 1))


def log_sum_exp(x):
    """Computes log_sum_exp in a numerically stable way."""

    # shape of x: [batch_size*K, seq_len, num_logistic]
    c = torch.max(x, dim=-1, keepdim=True)
    val = torch.squeeze(c.values, dim=-1) + torch.log(torch.sum(torch.exp(x - c.values), dim=-1))
    return val


def train_parallel_wavenet(teacher_wavenet: WavenetModel, 
                           student_wavenet: WavenetModel,
                           num_logistic: int,
                           model_name: str):
    """Implements distillation training loop for parallel (student) wavenet."""

    num_epochs = 50
    seq_len = 150000
    K = 10
    epsilon = 1e-8
    batch_size = 1

    teacher_wavenet.eval()
    student_wavenet.train()

    dataset = ParallelWavenetDataset(seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    optimizer = torch.optim.Adam(student_wavenet.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for batch, z in enumerate(dataloader):
            optimizer.zero_grad()
            
            # generate sample from student
            pre_out = student_wavenet(z)                          # (batch_size, out_channels, seq_len)
            stu_loc = pre_out[:, 0, :]                            # (batch_size, seq_len)
            stu_s = F.softplus(pre_out[:, 1, :])                  # (batch_size, seq_len)
            stu_loc = torch.unsqueeze(stu_loc, dim=-1)            # (batch_size, seq_len, 1)
            stu_s = torch.unsqueeze(stu_s, dim=-1)                # (batch_size, seq_len, 1)

            stu_loc_repeated = stu_loc.repeat(1, 1, K)            # (batch_size, seq_len, K)
            stu_s_repeated = stu_s.repeat(1, 1, K)                # (batch_size, seq_len, K)

            # Get K samples for each timestep t from student to estimate cross-entropy b/w student and teacher
            uniform_random = torch.rand(stu_loc_repeated.shape) + epsilon              # (batch_size, seq_len, K)
            temp = torch.log(uniform_random / (1 - uniform_random))                    # (batch_size, seq_len, K)
            stu_x = stu_loc_repeated +  stu_s_repeated * temp                          # (batch_size, seq_len, K)

            # get loc,scale estimates from teacher for the generated samples
            stu_x_transposed = stu_x.transpose(1, 2)                                                  # (batch_size, K, seq_len)
            stu_x_transposed = torch.reshape(stu_x_transposed, (batch_size*K, seq_len, 1))            # (batch_size * K, seq_len, 1)
            tch_pre_out = teacher_wavenet(stu_x_transposed)                                           # (batch_size * K, num_logistic*3, seq_len)
            
            tch_phi_logits = tch_pre_out[:, :num_logistic, :]                 # (batch_size * K, num_logistic, seq_len)
            tch_phi = F.softmax(tch_phi_logits, dim=1)                        # (batch_size * K, num_logistic, seq_len)
            tch_mu = tch_pre_out[:, num_logistic:2*num_logistic, :]           # (batch_size * K, num_logistic, seq_len)
            tch_s = F.softplus(tch_pre_out[:, 2*num_logistic:, :])            # (batch_size * K, num_logistic, seq_len)

            tch_phi_t = tch_phi.transpose(1, 2)                      # (batch_size * K, seq_len, num_logistic)
            tch_mu_t = tch_mu.transpose(1, 2)                        # (batch_size * K, seq_len, num_logistic)
            tch_s_t = tch_s.transpose(1, 2)                          # (batch_size * K, seq_len, num_logistic)

            # calculate cross entropy b/w student and teacher dist
            stu_x_t_r = stu_x_transposed.repeat(1, 1, num_logistic)     # (batch_size * K, seq_len, num_logistic)
            t1 = -((stu_x_t_r - tch_mu_t) / (tch_s_t + epsilon))        # (batch_size*K, seq_len, num_logistic)
            t2 = -torch.log(tch_s_t + epsilon)                          # (batch_size*K, seq_len, num_logistic)                     
            t3 = -2 * F.softplus(t1)
            log_phi = torch.log(tch_phi_t)                              # (batch_size*K, seq_len, num_logistic)
            t4 = t1 + t2 + t3 +  log_phi                                # (batch_size*K, seq_len, num_logistic)
            t5 = log_sum_exp(t4)                                        # (batch_size*K, seq_len)
            t6 = torch.reshape(t5, (batch_size, K, seq_len))            # (batch_size, K, seq_len)
            t7 = t6.transpose(1, 2)                                     # (batch_size, seq_len, K)
            
            t8 = torch.mean(t7, dim=-1)                                                # (batch_size, seq_len)
            t9 = torch.sum(t8, dim=1)                                                  # (batch_size)

            # calculate entropy of student distribution
            t10 = torch.log(torch.squeeze(stu_s, dim=-1) + epsilon)                     # (batch_size, seq_len)
            t11 = torch.sum(t10, dim=-1)                                                # (batch_size)
            t12 = -t9 - t11
            loss = torch.mean(t12, dim=0)

            loss.backward()
            optimizer.step()
            print(f'epoch: {epoch}, batch: {batch}, loss: {loss.item()}')

        # Save the model every epoch
        torch.save(student_wavenet.state_dict(), model_name)


def sample_parallel_wavenet(student_wavenet: WavenetModel, model_path):
    gen_len = 20000
    student_wavenet.load_state_dict(torch.load(model_path, weights_only=True))

    # generate noise input
    z_dist = torch.distributions.TransformedDistribution(
        torch.distributions.Uniform(0, 1),
        [torch.distributions.transforms.SigmoidTransform().inv,
         torch.distributions.transforms.AffineTransform(loc=0.0, scale=1.0)]
         )
    z = z_dist.sample((1, gen_len, 1))               # (batch_size, gen_len, in_channels)
    pre_out = student_wavenet(z)                     # (batch_size, out_channels, gen_len)
    mu = pre_out[0, 0, :]                            # (gen_len)
    s = F.softplus(pre_out[0, 1, :])                 # (gen_len)

    x_dist = torch.distributions.TransformedDistribution(
        torch.distributions.Uniform(0, 1),
        [torch.distributions.transforms.SigmoidTransform().inv,
         torch.distributions.transforms.AffineTransform(loc=mu, scale=s)]
         )
    
    x = x_dist.sample()
    x = torch.ceil(x)
    x = torch.where(x > 255.0, 255.0, x)
    x = torch.where(x < 0.0, 0.0, x)

    wavfile.write('sampled_output_logistic.wav', rate=22050, data=x.numpy().astype(np.int16))
    print('sampling complete !')

        


def main():
    num_logistic = 5

    # Create teacher Wavenet config.
    # Wavenet model will output num_logistic * (phi, u, s) for each output audio sample (x_t)
    teacher_config = WavenetConfig(
        rb_c1_kernel_size = 2,
        rb_c1_in_channels = 16,
        rb_c1_out_channels = 32,

        rb_c2_in_channels = 16,
        rb_c2_out_channels = 16,

        ol_c1_in_channels = 16,
        ol_c1_out_channels = 16,

        ol_c2_in_channels = 16,
        ol_c2_out_channels = (num_logistic * 3),

        # 10 dilation layer will mean one complete dilation of [1, 2, 4, 8, ..., 512]
        num_residual_blocks = 40, # check if this needs to be increased

        il_kernel_size = 2,
        il_in_channels = 1,
        il_out_channels = 16
    )

    trained_model_name = 'wavenet_logistic_mix_trained.pt'
    teacher_wavenet = WavenetModel(teacher_config)
    teacher_wavenet.load_state_dict(torch.load(trained_model_name, weights_only=True))

    # Create student Wavenet config.
    # For student, we use only a single logistic distribution for the mixture.
    # Wavenet model will output (phi, u, s) for each output audio sample (x_t)
    student_config = WavenetConfig(
        rb_c1_kernel_size = 2,
        rb_c1_in_channels = 16,
        rb_c1_out_channels = 32,

        rb_c2_in_channels = 16,
        rb_c2_out_channels = 16,

        ol_c1_in_channels = 16,
        ol_c1_out_channels = 16,

        ol_c2_in_channels = 16,
        ol_c2_out_channels = 2,

        # 10 dilation layer will mean one complete dilation of [1, 2, 4, 8, ..., 512]
        num_residual_blocks = 40, # check if this needs to be increased

        il_kernel_size = 2,
        il_in_channels = 1,
        il_out_channels = 16
    )

    student_model_name = 'wavenet_student_logistic_mix_trained.pt'
    student_wavenet = WavenetModel(student_config)
    train_parallel_wavenet(teacher_wavenet, student_wavenet, num_logistic, student_model_name)
    sample_parallel_wavenet(student_wavenet, student_model_name)



if __name__ == "__main__":
    main()
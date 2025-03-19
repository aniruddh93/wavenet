# training code for wavenet model
# Implements p(x_t/x_1, ..., X_t-1) in two different ways:
# (a) 256-Categorical with Softmax 
# (b) Mixture of logistic distribution

import os
import torch
from scipy.io import wavfile
import csv
import numpy as np
import sys

sys.path.append('/kaggle/input/wavenet_base_model/pytorch/default/1/')
from wavenet_model import WavenetConfig, WavenetModel

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class WaveNetDataset(Dataset):
    """Provides training and test data """

    def __init__(self, dataset_path: str, train: bool, seq_len: int, shift: float = 0.0):
        super().__init__()
        self.dataset_path = dataset_path
        metadata_csv_path = os.path.join(dataset_path, 'metadata.csv')
        self.wav_files = []
        self.seq_len = seq_len
        self.shift_val = shift

        with open(metadata_csv_path, 'r') as f:
            csv_reader =  csv.reader(f)

            for row in csv_reader:
                t_str = ''.join(row)
                parts = t_str.split('|')
                self.wav_files.append(parts[0])

                if len(self.wav_files) >= 32:
                    return

    def __len__(self):
        return len(self.wav_files)
    
    def __getitem__(self, index):
        wav_file = self.wav_files[index] + '.wav'
        wav_file_path = os.path.join(self.dataset_path, 'wavs', wav_file)
        _, wav_data = wavfile.read(wav_file_path)
        
        # apply transformations
        wav_tensor = self.normalize(wav_data)
        wav_tensor = self.u_law_transform(wav_tensor)
        wav_tensor, masks = self.quantize(wav_tensor)
        wav_tensor, masks = self.pad_sequence(wav_tensor, masks)
        wav_tensor, masks =  self.shift_sequence(wav_tensor, masks)
        return wav_tensor, masks


    def normalize(self, wav_data) -> torch.Tensor:
        """Converts numpy input data to PyTorch tensor and Normalizes data to be between 1 & -1."""
        wav_tensor = torch.Tensor(wav_data)
        wav_tensor = wav_tensor / (2**15)
        wav_tensor = torch.clip(wav_tensor, min=-1.0, max=1.0)
        return wav_tensor

    def u_law_transform(self, wav_tensor):
        """
        Applies u-law companding transformation with u=255.
        """
        t1 = 1 + (255 * torch.abs(wav_tensor))
        wav_tensor = torch.sign(wav_tensor) * torch.log2(t1)
        wav_tensor = wav_tensor / 8
        return wav_tensor
    
    def quantize(self, wav_tensor):
        """Converts input tensor values to the closest quantized values.
        Also returns mask corresponding to the quantized value that was selected."""

        quantized_values = [i*(2.0/256.0) for i in range(-128, 128, 1)]
        input_tensor_copy = torch.clone(wav_tensor)

        for i in range(len(quantized_values) - 1):
            wav_tensor = torch.where((wav_tensor >= quantized_values[i]) & (wav_tensor <= quantized_values[i+1]),
                                     quantized_values[i], wav_tensor)

        for i in range(len(quantized_values) - 2, -1, -1):
            input_tensor_copy = torch.where((input_tensor_copy >= quantized_values[i]) & (input_tensor_copy <= quantized_values[i+1]),
                                            i, input_tensor_copy)
            
        input_tensor_copy = input_tensor_copy.to(torch.int64)
        input_tensor_copy = torch.unsqueeze(input_tensor_copy, dim=-1)

        assert len(wav_tensor.shape) == 1

        masks = torch.zeros(wav_tensor.shape[0], 256, dtype=torch.float32)
        values = torch.ones(wav_tensor.shape[0], 1, dtype=torch.float32)
        masks.scatter_(dim=-1, index=input_tensor_copy, src=values)
        return wav_tensor, masks
    
    def pad_sequence(self, wav_tensor, masks):
        """Pads/cuts-off the seq to self.seq_len"""

        if wav_tensor.shape[0] < self.seq_len:
            # pad it with "0.0" which is a quantized value
            # also, adjust the mask so loss function is not applied for padding

            padded_wav_tensor = torch.zeros(self.seq_len, dtype=torch.float32)
            padded_wav_tensor[:wav_tensor.shape[0]] = wav_tensor[:]

            padded_masks = torch.zeros(self.seq_len, 256, dtype=torch.float32)
            padded_masks[:masks.shape[0], :] = masks[:, :]
            return padded_wav_tensor, padded_masks
        
        else:
            # slice the wav_tensor & mask to seq_len
            sliced_wav_tensor = wav_tensor[:self.seq_len]
            sliced_masks = masks[:self.seq_len, :]
            return sliced_wav_tensor, sliced_masks
        
    def shift_sequence(self, wav_tensor, masks):
        wav_tensor = wav_tensor - self.shift_val
        return wav_tensor, masks


def train_categorical(config: WavenetConfig, model: WavenetModel, model_name: str):
    """Implements the training routine when p(x_t/x_1..x_t-1) is modeled as
       a categorical (softmax) distribution."""
    
    model.train()
    print('Successfully created the model !')

    num_epochs = 50
    dataset_path = '/kaggle/input/the-lj-speech-dataset/LJSpeech-1.1'
    seq_len = 150000
    losses = []

    train_dataset = WaveNetDataset(dataset_path, train=True, seq_len=seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    loss = nn.CrossEntropyLoss()
    print('loaded dataset, starting training !')

    for epoch in range(num_epochs):
        for batch, (wav_tensor, mask) in enumerate(train_dataloader):     
            # add channel dim
            wav_tensor = torch.unsqueeze(wav_tensor, dim=-1)
            pre_out = model(wav_tensor)

            # pre_out shape: (batch_size, out_channels, seq_len)
            # mask shape: (batch_size, seq_len, out_channels)
            # reshape mask to (batch_size, out_channels, seq_len) for loss calculation
            mask = torch.transpose(mask, 1, 2)
            l = loss(pre_out, mask)
            losses.append(l.item())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            # if batch % 10 == 0:
            print(f'epoch: {epoch}, batch: {batch}, loss: {l.item()}')

        # Save the model every epoch
        torch.save(model.state_dict(), model_name)


def sample_categorical(model, max_gen_len, model_name):
    """Samples from the trained model where p(x_t/x_1..x_t-1) is modeled as
       a categorical (softmax) distribution."""
    
    # load the trained model
    model.load_state_dict(torch.load(model_name, weights_only=True))
    model.eval()

    # sample some random initial data
    data = torch.randint(low=1, high=127, size=(1, 1)).to(torch.float32)
    data = torch.unsqueeze(data, dim=-1)
    
    for i in range(max_gen_len):
        pre_out = model(data)          # [batch_size, out_channels, seq_len]
        out = pre_out[:, :, -1]        # [batch_size, out_channels]
        out = torch.max(out, dim=-1)

        step_output = out.indices                                    # [batch_size]
        step_output =  torch.unsqueeze(step_output, dim=-1)          # [batch_size, seq_len]
        step_output =  torch.unsqueeze(step_output, dim=-1)          # [batch_size, seq_len, num_channels]
        data = torch.cat((data, step_output), dim=1)

        if i % 100 == 0:
            print(i)

    print(data.shape)
    print(data.numpy().astype(np.int16))
    wavfile.write('sampled_output_categorical.wav', rate=22050, data=data[0, :, 0].numpy().astype(np.int16))


def log_sum_exp(x):
    """Computes log_sum_exp in a numerically stable way."""

    # shape of x: [batch_size, num_logistic, seq_len]
    c = torch.max(x, dim=1, keepdim=True)
    val = torch.squeeze(c.values, dim=1) + torch.log(torch.sum(torch.exp(x - c.values), dim=1))
    return val


def log_mix_logistic(x, phi_logits, u, s):
    """Implements log mixture of logistic."""

    num_logistic = phi_logits.shape[1]
    x = torch.transpose(x, 1, 2)         # [batch_size, in_channels, seq_len]

    # replicate x so in_channels becomes equal to num_logistic
    x_expanded = x.repeat(1, num_logistic, 1)
    t1 = F.log_softmax(phi_logits, dim=1)
    t2 = -((x_expanded - u) / s)
    t3 = t2 - torch.log(s) - 2*torch.log(1 + torch.exp(t2))
    t4 = t1 + t3
    t5 = log_sum_exp(t4)          # [batch_size, seq_len]
    t6 = torch.sum(t5, dim=-1)    # [batch_size]
    loss = -torch.mean(t6)
    return loss


def train_logistic_mix(config: WavenetConfig, model: WavenetModel, model_name: str, num_logistic: int):
    """Implements the training routine when p(x_t/x_1..x_t-1) is modeled as
       a mixture of logistic distribution."""
    
    model.train()
    print('Successfully created the model !')

    num_epochs = 50
    dataset_path = '/kaggle/input/the-lj-speech-dataset/LJSpeech-1.1'
    seq_len = 150000
    
    # dataset, dataloader, optimizer, loss
    train_dataset = WaveNetDataset(dataset_path, train=True, seq_len=seq_len, shift=0.5)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

    for epoch in range(num_epochs):
        for batch, (wav_tensor, mask) in enumerate(train_dataloader):
            wav_tensor = torch.unsqueeze(wav_tensor, dim=-1)        # [batch_size, seq_len, in_channels]
            pre_out = model(wav_tensor)                             # [batch_size, out_channels, seq_len]
            phi_logits = pre_out[:, 0:num_logistic, :]              # [batch_size, num_logistic, seq_len]
            u = pre_out[:, num_logistic:2*num_logistic, :]          # [batch_size, num_logistic, seq_len]
            s = F.softplus(pre_out[:, 2*num_logistic:, :])          # [batch_size, num_logistic, seq_len]

            loss = log_mix_logistic(wav_tensor, phi_logits, u, s)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batch % 10 == 0:
            print(f'epoch: {epoch}, batch: {batch} | loss: {loss.item()}')

        # Save model every epoch
        torch.save(model.state_dict(), model_name)


def sample_logistic_mix(model, num_logistic, max_gen_len, model_name):
    """Samples from the trained model where p(x_t/x_1..x_t-1) is modeled as
       a mixture of logistic distribution."""
    
    model.load_state_dict(torch.load(model_name, weights_only=True))
    model.eval()

     # sample some random initial data
    data = torch.randint(low=1, high=127, size=(1, 20)).to(torch.float32)
    data = torch.unsqueeze(data, dim=-1)

    # Shift by 0.5 since we are modeling discretized mixture of logistics
    data = data - 0.5

    for i in range(max_gen_len):
        pre_out = model(data)          # [batch_size, out_channels, seq_len]
        out = pre_out[:, :, -1]        # [batch_size, out_channels]

        phi_logits = out[:, :num_logistic]
        phi = F.softmax(phi_logits, dim=-1)
        u = out[:, num_logistic:2*num_logistic]
        s = F.softplus(out[:, 2*num_logistic:])

        # To sample from mixture of logits, sample a logistic distribution
        # and then sample data from this logistic distribution
        sampled_logistic_idx = torch.multinomial(phi, num_samples=1, replacement=True)
        sampled_u = torch.squeeze(u[:, sampled_logistic_idx])
        sampled_s = torch.squeeze(s[:, sampled_logistic_idx])
        dist = torch.distributions.TransformedDistribution(
            torch.distributions.Uniform(0, 1),
            [torch.distributions.transforms.SigmoidTransform().inv, 
             torch.distributions.transforms.AffineTransform(loc=sampled_u, scale=sampled_s)]
        )

        sampled_x = dist.sample((1, 1, 1))
        sampled_x = torch.ceil(sampled_x)
        sampled_x = torch.where(sampled_x > 255.0, 255.0, sampled_x)
        sampled_x = torch.where(sampled_x < 0.0, 0.0, sampled_x)

        data = torch.cat((data, sampled_x), dim=1)
        if i % 100 == 0:
            print(f'step {i}')

    print(data.shape)
    print(data.numpy().astype(np.int16))
    wavfile.write('sampled_output_logistic.wav', rate=22050, data=data[0, :, 0].numpy().astype(np.int16))


def main():
    modelling_type = "logistic_mixture"

    if modelling_type ==  "categorical_softmax":
        # Create Wavenet config
        config = WavenetConfig(
            rb_c1_kernel_size = 2,
            rb_c1_in_channels = 16,
            rb_c1_out_channels = 32,

            rb_c2_in_channels = 16,
            rb_c2_out_channels = 16,

            ol_c1_in_channels = 16,
            ol_c1_out_channels = 16,

            ol_c2_in_channels = 16,
            ol_c2_out_channels = 256,

            # 10 dilation layer will mean one complete dilation of [1, 2, 4, 8, ..., 512]
            num_residual_blocks = 40, # check if this needs to be increased

            il_kernel_size = 2,
            il_in_channels = 1,
            il_out_channels = 16
        )

        model_name = 'wavenet_categorical_trained.pt'

        model = WavenetModel(config)
        train_categorical(config, model, model_name)
        sample_categorical(model, max_gen_len=20000, model_name=model_name)

    elif modelling_type == "logistic_mixture":
        num_logistic = 5

        # Create Wavenet config.
        # Wavenet model will output num_logistic * (phi, u, s) for each output audio sample (x_t)
        config = WavenetConfig(
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

        model_name = 'wavenet_logistic_mix_trained.pt'

        model = WavenetModel(config)
        # train_logistic_mix(config, model, model_name, num_logistic)
        sample_logistic_mix(model, num_logistic, max_gen_len=20000, model_name=model_name)


if __name__ == "__main__":
    main()
 
# Implementation of Wavenet model as described in the paper.

import os
import csv
from scipy.io import wavfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

@dataclass
class WavenetConfig:
    """Config knobs for wavenet model."""

    # Config for conv1 (dilated causal conv) in residual blocks
    rb_c1_kernel_size: int
    rb_c1_in_channels: int
    rb_c1_out_channels: int

    # Config for conv2 (1x1 conv) in residual blocks
    rb_c2_in_channels: int
    rb_c2_out_channels: int

    # Config for conv1 (1x1 conv) in output layer
    ol_c1_in_channels: int
    ol_c1_out_channels: int

    # Config for conv2 (1x1 conv) in output layer
    ol_c2_in_channels: int
    ol_c2_out_channels: int

    # num of residual block layers in Wavenet model
    num_residual_blocks: int

    # Config for input layer causal conv
    il_kernel_size: int
    il_in_channels: int
    il_out_channels: int



class Causal1DConv(nn.Module):
    """Implements 1-D causal convolution"""

    def __init__(
            self,
            layer_dilation_factor: int,
            kernel_size: int,
            in_channels: int,
            out_channels: int) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * layer_dilation_factor

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=layer_dilation_factor,
            padding=self.padding,
            bias=True
        )

    def forward(self, input):
        result = self.conv(input)    # (batch_size, out_channels, seq_len)

        if self.padding > 0:
            return result[:, :, :-self.padding]
        
        return result


class ResidualBlock(nn.Module):
    """Implements a residual block (Sec. 2.4 in paper)."""

    def __init__(self, layer_id: int, config: WavenetConfig):
        super().__init__()

        self.layer_dilation_factor = 2 ** (layer_id % 10)
        self.causal_dilated_conv = Causal1DConv(
            self.layer_dilation_factor,
            config.rb_c1_kernel_size,
            config.rb_c1_in_channels,
            config.rb_c1_out_channels
        )
        
        self.conv2 = nn.Conv1d(
            config.rb_c2_in_channels,
            config.rb_c2_out_channels,
            kernel_size=1,
            padding=0,
            bias=True
        )

        self.config = config


    def forward(self, input, out_list: list):
        dc_out = self.causal_dilated_conv(input)

        # split output along channel dimension (one part for tanh, other for sigmoid)
        sub_tensors_dc_out = torch.split(dc_out, self.config.rb_c2_in_channels, dim=1)
        intermediate_res = F.tanh(sub_tensors_dc_out[0]) * F.sigmoid(sub_tensors_dc_out[1])
        
        out1 = self.conv2(intermediate_res)
        out_list.append(out1)
        out2 = out1 + input
        return out2
    

class OutputLayer(nn.Module):
    """Implements output layer in Wavenet model (Sec. 2.4 in paper)."""

    def __init__(self, config: WavenetConfig) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            config.ol_c1_in_channels,
            config.ol_c1_out_channels,
            kernel_size=1,
            bias=True
        )

        self.conv2 = nn.Conv1d(
            config.ol_c2_in_channels,
            config.ol_c2_out_channels,
            kernel_size=1,
            bias=True
        )


    def forward(self, input):
        t1 = F.relu(input)               # (batch_size, out_channels, seq_len)
        t2 = self.conv1(t1)              # (batch_size, out_channels, seq_len)
        t3 = F.relu(t2)                  # (batch_size, out_channels, seq_len)
        t4 = self.conv2(t3)              # (batch_size, out_channels, seq_len)
        out = F.softmax(t4, dim=1)       # (batch_size, out_channels, seq_len)

        return out, t4

    
class WavenetModel(nn.Module):
    """Implements the WaveNet model."""

    def __init__(self, config: WavenetConfig) -> None:
        super().__init__()
        self.config = config

        self.residual_blocks = []
        for i in range(config.num_residual_blocks):
            self.residual_blocks.append(ResidualBlock(i, config))

        self.output_layer = OutputLayer(config)
        self.input_causal_layer = Causal1DConv(
            layer_dilation_factor = 1,
            kernel_size = config.il_kernel_size,
            in_channels = config.il_in_channels,
            out_channels = config.il_out_channels)
        
    def forward(self, input):
        # input shape: (batch_size, seq_len, in_channels)
        # Conv1D expects input to be [batch, in_channels, seq_len]
        input = torch.transpose(input, 1, 2)
        
        o1 = self.input_causal_layer(input)
        intermediate_outputs = []

        for rb in self.residual_blocks:
            o1 = rb(o1, intermediate_outputs)

        summed_outputs = torch.zeros_like(intermediate_outputs[0], requires_grad=True)
        for io in intermediate_outputs:
            summed_outputs = summed_outputs + io

        out, pre_out = self.output_layer(summed_outputs)
        return out, pre_out


class WaveNetDataset(Dataset):
    """Provides training and test data """

    def __init__(self, dataset_path: str, train: bool, seq_len: int):
        super().__init__()
        self.dataset_path = dataset_path
        metadata_csv_path = os.path.join(dataset_path, 'metadata.csv')
        self.wav_files = []
        self.seq_len = seq_len

        with open(metadata_csv_path, 'r') as f:
            csv_reader =  csv.reader(f)

            for row in csv_reader:
                t_str = ''.join(row)
                parts = t_str.split('|')
                self.wav_files.append(parts[0])

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

        quantized_values = [i*(2.0/256.0) for i in range(-128, 129, 1)]
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


def train(config: WavenetConfig):
    """Implements the training routine."""

    model = WavenetModel(config)
    model.train()
    print('Successfully created the model !')

    num_epochs = 50
    dataset_path = '/kaggle/input/the-lj-speech-dataset/LJSpeech-1.1'
    seq_len = 150000
    losses = []

    train_dataset = WaveNetDataset(dataset_path, train=True, seq_len=seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    loss = nn.CrossEntropyLoss()
    print('loaded dataset, starting training !')

    for epoch in range(num_epochs):
        for batch, (wav_tensor, mask) in enumerate(train_dataloader):     
            # add channel dim
            wav_tensor = torch.unsqueeze(wav_tensor, dim=-1)
            out, pre_out = model(wav_tensor)

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



def main():
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
        num_residual_blocks = 20, # check if this needs to be increased

        il_kernel_size = 2,
        il_in_channels = 1,
        il_out_channels = 16
    )

    train(config)


if __name__ == "__main__":
    main()




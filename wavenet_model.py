# Implementation of Wavenet model as described in the paper.

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

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
        return t4

    
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

        pre_out = self.output_layer(summed_outputs)
        return pre_out

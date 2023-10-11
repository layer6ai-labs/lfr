import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class CustomizedLayerNorm(nn.Module):
    """
    Customized layer normalization to enable nomalization on specific dimension.
    """

    def __init__(self, normalized_shape: int, dim: int = -1):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.dim = dim
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        out = self.layer_norm(x)
        out = torch.transpose(out, self.dim, -1)
        return out


# Adapted from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py.
# In order to match the TCN in NCL, add layer normalization to replace the weight_norm, and
# slice the last piece of output in time dimension.
class Chomp1d(nn.Module):
    """
    Removing extra padding at the end of the sequence.
    """

    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self,  n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float=0.2, norm='layer_norm'):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if norm == "weight_norm":
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.relu2, self.dropout2)
            
        else:
            self.norm1 = CustomizedLayerNorm(n_outputs, 1)
            self.norm2 = CustomizedLayerNorm(n_outputs, 1)
            self.net = nn.Sequential(self.conv1, self.chomp1, self.norm1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.norm2, self.relu2, self.dropout2)
        

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, input_seq_len:int, num_channels: list, kernel_size: int=2, dropout: float=0.2, norm='layer_norm'):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, norm=norm)]

        self.network = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_channels[-1] * input_seq_len, num_channels[-1])

    def forward(self, x):
        out = self.network(x)
        out = self.flatten(out)
        out = self.linear(out)
        return out


class MirrorTemporalBlock(nn.Module):
    def __init__(self,  n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float=0.2, norm='layer_norm'):
        super(MirrorTemporalBlock, self).__init__()
        self.conv1 = nn.ConvTranspose1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.ConvTranspose1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if norm == "weight_norm":
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)
            self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                     self.conv2, self.relu2, self.dropout2)
            
        else:
            self.norm1 = CustomizedLayerNorm(n_outputs, 1)
            self.norm2 = CustomizedLayerNorm(n_outputs, 1)
            self.net = nn.Sequential(self.conv1, self.norm1, self.relu1, self.dropout1,
                                     self.conv2, self.norm2, self.relu2, self.dropout2)

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        return out


class MirrorTemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_channels: list, kernel_size: int=3, dropout: float=0.0, norm='layer_norm'):
        super(MirrorTemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [MirrorTemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=0, dropout=dropout, norm=norm)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out

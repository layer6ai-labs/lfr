import torch
import torch.nn as nn
from ssl_models.models.TCN import TemporalConvNet


class HARCnnTarget(nn.Module):
    def __init__(self, dim=512, input_channel=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, 16, kernel_size=8,
                stride=1, bias=False, padding=(8//2)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(16, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        
        self.mlp = nn.Sequential(
                    nn.Linear(32*34, 256),
                    nn.Linear(256, dim)
                    )

    def forward(self, xb):
        # Flatten images into vectors
        out = self.conv(xb)
        out = out.view(out.size(0), -1)
        out = self.mlp(out)
        return out


class EpilepsyCnnTarget(nn.Module):
    def __init__(self, dim=512, input_channel=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, 16, kernel_size=8,
                stride=1, bias=False, padding=(8//2)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(16, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        
        self.mlp = nn.Sequential(
                    nn.Linear(32*46, 256),
                    nn.Linear(256, dim)
                    )

    def forward(self, xb):
        # Flatten images into vectors
        out = self.conv(xb)
        out = out.view(out.size(0), -1)
        out = self.mlp(out)
        return out

    
class KvasirCnnTarget(nn.Module):
    def __init__(self, dim, layers=3):
        super().__init__()
        self.dim = dim
        self.layers = layers
        if layers == 1:
            self.network = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 16 x 16 x 16
                nn.Flatten(), 
                nn.Linear(32000, dim))
        elif layers == 2:
            self.network = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 16 x 16 x 16

                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 32 x 8 x 8

                nn.Flatten(), 
                nn.Linear(16000, dim))
        else:
            self.network = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 16 x 16 x 16

                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 32 x 8 x 8

                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 32 x 8 x 8

                nn.Flatten(), 
                nn.Linear(3840, dim))


    def forward(self, xb):
        out = self.network(xb)
        return out


class MIMIC3TcnTarget(nn.Module):
    def __init__(self, input_dim: int, input_seq_len: int, output_channels: list=[64,64,64], output_dim=64, kernel_size: int=2, 
                 dropout: float=0.0):
        super(MIMIC3TcnTarget, self).__init__()
        self.model = TemporalConvNet(input_dim, input_seq_len, output_channels, kernel_size, dropout)
        self.linear = nn.Linear(output_channels[-1], output_dim)
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        out = self.model(x)
        out = self.linear(out)
        return out

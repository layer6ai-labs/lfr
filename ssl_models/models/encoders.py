import torch
import torch.nn as nn
from ssl_models.models.TCN import TemporalConvNet
import torchvision


# reference: https://openreview.net/pdf?id=EfR55bFcrcI
class Simple4LayerMLP(nn.Module):
    def __init__(self, input_dim, dim=128):
        super().__init__()
        # 4-layer MLP
        self.mlp = nn.Sequential(nn.Linear(input_dim, 256), 
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 256), 
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 256), 
                                nn.ReLU(inplace=True),
                                nn.Linear(256, dim))
        
    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1)
        out = self.mlp(out)
        return out


class Simple2LayerMLP(nn.Module):
    def __init__(self, input_dim, dim=128):
        super().__init__()
        # 2-layer MLP
        self.mlp = nn.Sequential(nn.Linear(input_dim, 256), 
                                nn.ReLU(inplace=True),
                                nn.Linear(256, dim))
        
    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1)
        out = self.mlp(out)
        return out

#adapted from https://github.com/emadeldeen24/TS-TCC/blob/main/models/model.py
class HARSCnnEncoder(nn.Module):
    def __init__(self, dim=128, input_channel=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, 32, kernel_size=8,
                stride=1, bias=False, padding=(8//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        if dim == 128*18:
            self.mlp = nn.Identity()
        else:
            # use a linear layer to reach the latent shape
            self.mlp = nn.Linear(128*18, dim)
        

    def forward(self, xb):
        # Flatten images into vectors
        out = self.conv(xb)
        out = out.view(out.size(0), -1)
        out = self.mlp(out)
        return out


class EpilepsyCnnEncoder(nn.Module):
    def __init__(self, dim=128, input_channel=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, 32, kernel_size=8,
                stride=1, bias=False, padding=(8//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        
        if dim == 24*128:
            self.mlp = nn.Identity()
        else:
            # use a linear layer to reach the latent shape
            self.mlp = nn.Linear(3072, dim)

    def forward(self, xb):
        # Flatten images into vectors
        out = self.conv(xb)
        out = out.view(out.size(0), -1)
        out = self.mlp(out)
        return out


class KvasirCnnEncoder(nn.Module):
    def __init__(self, dim, input_channel=3):
        super().__init__()
        self.input_channel = input_channel
        self.conv = self._make_layers(
            [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.mlp = nn.Linear(3072, dim)
        
    def _make_layers(self, cfg):
        layers = []
        in_channel = self.input_channel
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channel, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channel = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, xb):
        out = self.conv(xb)
        out = out.view(out.size(0), -1)
        out = self.mlp(out)
        return out

class KvasirResnetEncoder(nn.Module):
    def __init__(self, arch, dim, args):
        super(KvasirResnetEncoder, self).__init__()
        base_encoder = torchvision.models.__dict__[args.arch]
        # Replace 1000 class output with dim class
        self.model = base_encoder(num_classes=dim, zero_init_residual=True)
        if 'supervised' not in args.method:
            self.model.fc = nn.Sequential(self.model.fc, nn.BatchNorm1d(dim, affine=False)) # output layer

    def forward(self, x):
        out = self.model(x)
        return out


# adapted from https://github.com/ratschlab/ncl/blob/main/model/architectures/encoders.py.
class MIMIC3TcnEncoder(nn.Module):
    """
    Sequence encoder based on TCN architectures from https://arxiv.org/pdf/1803.01271.pdf
    """
    def __init__(self, input_dim: int, input_seq_len: int, num_channels: list = [64,64,64,64,64], kernel_size: int = 2, 
                 dropout: float = 0.0, static_dropout: float = 0.5, l2_norm: bool=False, n_static: int = 1, 
                 embedding_size: int = None):
        super().__init__()

        self.TCN = TemporalConvNet(input_dim - n_static, input_seq_len, num_channels, kernel_size, dropout)
        self.n_static = n_static
        self.l2_norm = l2_norm
        self.static_dropout = nn.Dropout(static_dropout)
        self.dropout = nn.Dropout(dropout)
        self.embedding_size = num_channels[-1] if embedding_size is None else embedding_size
        self.FC_layer = nn.Linear(num_channels[-1] + n_static, self.embedding_size)
        self.layer_norm = nn.LayerNorm(self.embedding_size)

    def forward(self, x):
        """Forward pass function of the encoder.
        We split static features, first channel of the input, from the sequence of variables.
        We pass the sequence in the TCN block and merge both features using a Dense layer.
        We finally project the representation to the unit sphere as in https://arxiv.org/abs/1911.05722.
        Args:
            x: Tensor batch input. Shape: (batch_size, time_step, features)
        Returns:
            out : The batch of embeddings.
        """
        static, x = torch.split(x, [self.n_static, x.shape[-1] - self.n_static], dim=-1)
        x = torch.transpose(x, 1, 2) # output shape: (batch_size, channels, time_step)
        out = self.TCN(x)
        static = static[:, -1, :]
        static_out = self.static_dropout(static)
        out = torch.concat([static_out, out], dim=1)
        out = self.dropout(out)
        out = self.FC_layer(out)
        out = self.layer_norm(out)
        if self.l2_norm:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out

import torch
import torch.nn as nn
from ssl_models.models.TCN import MirrorTemporalConvNet


class HARCnnDecoder(nn.Module):
    def __init__(self, dim=100, input_channel=9):
        super().__init__()

        self.mlp = nn.Sequential(
                    nn.Linear(dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, 32*34)
                    )

        self.deconv = nn.Sequential(
                        nn.ConvTranspose1d(32, 16, kernel_size=8,
                            stride=1, bias=False, padding=4),
                        nn.BatchNorm1d(16),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2),
                        nn.ConvTranspose1d(16, input_channel, kernel_size=7, stride=1, bias=False, padding=4),
                        nn.BatchNorm1d(input_channel),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2),
                        nn.Flatten(),
                        nn.Linear(9*128, 9*128)
                        )
        

    def forward(self, xb):
        # Flatten images into vectors
        out = self.mlp(xb)
        out = out.view(out.size(0), 32, 34)
        out = self.deconv(out)
        return out.view((-1, 9, 128))


class EpilepsyCnnDecoder(nn.Module):
    def __init__(self, dim=100, input_channel=1):
        super().__init__()
        self.mlp = nn.Sequential(
                    nn.Linear(dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, 32*46)
                    )

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=8,
                stride=1, bias=False, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(16, input_channel, kernel_size=8, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(input_channel),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Flatten(),
            nn.Linear(178, 178),
            nn.ReLU()
        )
        
    def forward(self, xb):
        # Flatten images into vectors
        out = self.mlp(xb)
        out = out.view(out.size(0), 32, 46)
        out = self.deconv(out)
        return out.view((-1, 1, 178))
    

class KvasirCNNDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 3-layer MLP
        self.linear = nn.Linear(dim, 32*25*20, bias=False)
        self.convnet = nn.Sequential(
                        nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),            

                        nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.UpsamplingNearest2d(scale_factor=2),

                        nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(8, 3, kernel_size=3,  padding=1),
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.Flatten(),
                        nn.Linear(3*80*100, 3*80*100),
                        nn.Sigmoid())
        
    def forward(self, xb):
        # Flatten images into vectors
        out = self.linear(xb).view(-1, 32, 25, 20)
        out = self.convnet(out)
        return out.view((-1,3,80,100))
    

# Adapted from https://github.com/ratschlab/ncl/blob/main/model/architectures/decoders.py.
class MIMIC3TcnDecoder(nn.Module):
    """
    Sequence decoder based on TCN architectures.
    """
    def __init__(self, input_dim: int, seq_len: int, num_channels: list = [64,64,64,64,41], kernel_size: int = 3, 
                 dropout: float = 0.0, n_static: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.n_static = n_static
        self.fs_layer = nn.Linear(input_dim, input_dim + n_static)
        self.MTCN = MirrorTemporalConvNet(input_dim, num_channels, kernel_size, dropout)

    def forward(self, x):
        """Forward pass function of the decoder.
        We pass the embedding to a fully connected layer, 
        then split static features - the first n_static channels of the data.
        We pass the variable features into mirrored TCN network.
        Finally the static features are repeated for each time step and merged with variable ones.
        
        Args:
            x: Tensor batch input. Shape: (batch_size, features)
        Returns:
            out : The batch of reconstructed data.
        """
        x = torch.unsqueeze(x, 1)
        x = self.fs_layer(x)
        static, x = torch.split(x, [self.n_static, x.shape[-1] - self.n_static], dim=-1)
        x = torch.transpose(x, 1, 2) # output shape: (batch_size, channels, time_step)
        x = self.MTCN(x)
        x = torch.transpose(x, 1, 2) # output shape: (batch_size, channels, time_step)
        x = x[:, :self.seq_len, :]
        static = static.repeat(1, self.seq_len, 1)
        out = torch.concat((static, x), dim=-1)
        return out

import torch.nn as nn
from ssl_models.models.model_helper import build_encoder, build_decoder
from ssl_models.models.decoders import *


class AutoEncoder(nn.Module):
    """
    Build a model.
    """
    def __init__(self, device=None, args=None):
        super(AutoEncoder, self).__init__()
        self.online_encoder = build_encoder(args=args).to(device)
        self.decoder = build_decoder(args=args).to(device)

    def forward(self, x):
        z_w = self.online_encoder(x) # NxC
        x_recon = self.decoder(z_w)
        return x_recon

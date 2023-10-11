import torch
import torch.nn as nn
import torchvision 
from ssl_models.models.model_helper import build_encoder, build_decoder
from ssl_models.models.decoders import *


class Diet(nn.Module):
    """
    Build a model.
    """
    def __init__(self,  args=None, num_data=0):
        super(Diet, self).__init__()
        self.online_encoder = build_encoder(args=args)
        self.predictor = nn.Linear(args.dim, num_data)

    def forward(self, x):
        z_w = self.online_encoder(x) # NxC
        output = self.predictor(z_w)
        return z_w, output

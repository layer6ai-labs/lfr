# reference: https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/weight_drop.html
import torch
import torch.nn.functional as F
import torch.nn as nn

from ssl_models.models.model_helper import build_encoder, build_predictor, build_projector
from torch.nn import Parameter

def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = torch.nn.parameter.Parameter(torch.nn.functional.dropout(raw_w, p=dropout, training=module.training))
            setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)


class STab(nn.Module):
    """
    Build a STab model.
    """
    def __init__(self, dim=2048, pred_dim=512, args=None):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(STab, self).__init__()

        # create the encoder
        online_encoder = build_encoder(args)
        for m in online_encoder.mlp:
            if type(m) == torch.nn.Linear:
                 m = _weight_drop(m, ['weight'], args.stab_drop_rate)
        self.online_encoder = online_encoder
        
        self.projector = build_projector(dim, args.proj_dim, args)
        self.encoder = nn.Sequential(self.online_encoder, self.projector)

        # build a predictor
        self.predictor = build_predictor(dim, pred_dim, args)

    def forward(self, x):
        z1 = self.online_encoder(x)
        p1 = self.predictor(z1)

        z2 = self.online_encoder(x)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()
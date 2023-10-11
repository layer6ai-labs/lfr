import torch.nn as nn
from ssl_models.models.model_helper import build_encoder, build_predictor, build_projector

# adapted from https://github.com/facebookresearch/simsiam
class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, dim=2048, pred_dim=512, args=None):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        self.online_encoder = build_encoder(args)
        self.projector = build_projector(dim, args.proj_dim, args)
        self.encoder = nn.Sequential(self.online_encoder, self.projector)

        # build a predictor
        self.predictor = build_predictor(dim, pred_dim, args)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

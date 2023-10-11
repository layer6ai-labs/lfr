import torch
import torch.nn as nn
import torchvision 

from ssl_models.models.model_helper import build_encoder, build_target, build_predictor
from ssl_models.models.targets import *
from ssl_models.models.decoders import *
import torch.nn.functional as F
from utils.dpp import *


class LFR(nn.Module):
    """
    Build a model.
    """
    def __init__(self, dim=2048, pred_dim=512, num_targets=3, device=None, args=None, sample_data=None):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(LFR, self).__init__()

        self.num_targets=num_targets

        # create the encoder
        self.online_encoder = build_encoder(args=args).to(device)

        target_encoders=[]
        predictors=[]
        
        for i in range(num_targets):
            with torch.no_grad():
                # randomly initialize target networks with no gradient 
                for _ in range(args.target_sample_ratio):
                    target = build_target(args, args.target_layers[i], train_data=sample_data, device=device)
                    for param_k in target.parameters():
                        param_k.requires_grad = False  # not update by gradient
                    target_encoders.append(target)
            predictor = build_predictor(dim, pred_dim, args).to(device)
            predictors.append(predictor)
        
        if args.target_sample_ratio > 1:
            print("===============selecting {} target encoders from {}===============".format(num_targets, len(target_encoders)))
            target_encoders = self.select_targets(target_encoders, num_targets, sample_data)

        self.predictors = nn.ModuleList(predictors)
        self.target_encoders = nn.ModuleList(target_encoders)
        self.args=args


    def forward(self, x):
        """
        Input:
            x: input images
        Output:
            predicted_reps, target_reps: predictors and targets of the network
        """

        # compute features for online encoder
        z_w = self.online_encoder(x) # NxC

        target_reps = []
        predicted_reps = []
        for i in range(self.num_targets):
            target = self.target_encoders[i]
            predictor = self.predictors[i]
            z_a = target(x) # NxC
            p_a = predictor(z_w)
            target_reps.append(z_a)
            predicted_reps.append(p_a)
        return predicted_reps, target_reps

    def select_targets(self, target_encoders, num_targets, sample_data):
        '''
        select num_targets number of encoders out of target_encoders
        ''' 
        with torch.no_grad():
            sims = []
            for t in target_encoders:
                # (bs, dim)
                rep = t(sample_data)
                if rep.shape[0] > 1000: 
                    rep = rep[np.random.RandomState(seed=42).permutation(np.arange(rep.shape[0]))[:1000]]
                rep_normalized = F.normalize(rep, dim=1)
                # (bs, bs) cosine similarity
                sim = rep_normalized @ rep_normalized.T
                sims.append(sim.view(-1))
            # N, bs^2
            sims = torch.stack(sims)
            sims_normalized = F.normalize(sims, dim=1)
            # N,N
            sims_targets = sims_normalized @ sims_normalized.T
            result = dpp(sims_targets.cpu().numpy(), num_targets)
        return [target_encoders[idx] for idx in result]

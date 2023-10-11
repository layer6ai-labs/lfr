import numpy as np
import random
import torch
import warnings
warnings.filterwarnings('ignore')


def TabTransform(augmentation=True, contrastive=True):
    if not (augmentation or contrastive): return None
    return Corrupt(corruption_rate=0.6, contrastive=contrastive)

class Corrupt(object):
    """
        Augmentation corrupt
    """

    def __init__(self, corruption_rate=0.6, contrastive=False):
        self.corruption_rate = corruption_rate
        self.contrastive = contrastive

    def __call__(self, sample, random_sample):
        if not self.contrastive:
            if np.random.uniform() > 0.5: return sample
        num_feat = sample.shape[0]
        corruption_mask = torch.zeros_like(random_sample, dtype=torch.bool)
        corruption_idx = torch.randperm(num_feat)[: int(num_feat*self.corruption_rate)]
        corruption_mask[corruption_idx] = True
        corrupted = torch.where(corruption_mask, random_sample, sample)
        if self.contrastive: 
            return sample, corrupted
        return corrupted
        


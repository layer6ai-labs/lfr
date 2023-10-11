import math
from PIL import ImageFilter
import random
from utils.transformations import *


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_transforms(method, dataset, labelled):        
    if dataset in ['har', 'epilepsy', 'mimic3-los']:
        transforms_cls = TSTransform
    elif dataset in ['theorem', 'uci-income', 'hepmass']:
        transforms_cls = TabTransform
    else:
        return None, None
    if labelled:
        if method == 'supervised-aug':
            train_transforms = transforms_cls(augmentation=True, contrastive=False)
        else:
            train_transforms = transforms_cls(augmentation=False, contrastive=False)
        test_transforms = transforms_cls(augmentation=False, contrastive=False)
    else:
        if method in ['simclr', 'simsiam']:
            train_transforms = transforms_cls(augmentation=True, contrastive=True)
            test_transforms = transforms_cls(augmentation=True, contrastive=True)
        elif 'aug' in method:
            train_transforms = transforms_cls(augmentation=True, contrastive=False)
            test_transforms = transforms_cls(augmentation=False, contrastive=False)
        else:
            train_transforms = transforms_cls(augmentation=False, contrastive=False)
            test_transforms = transforms_cls(augmentation=False, contrastive=False)
    return train_transforms, test_transforms
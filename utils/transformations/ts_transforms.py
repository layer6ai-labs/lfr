import numpy as np
import random
import torch
from torchvision.transforms import Lambda
import warnings
warnings.filterwarnings('ignore')


def TSTransform(augmentation=True, contrastive=True):
    # https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb
    if not (augmentation or contrastive): return None
    if contrastive:
        return Lambda(lambda sample: (scaling(sample), jitter(permutation(sample))))
    else:
        return Lambda(lambda sample: random.choice([scaling(sample), jitter(permutation(sample))]))


def jitter(x, sigma=0.05):
    # https://arxiv.org/pdf/1706.00527.pdf
    new = x + np.random.normal(loc=0., scale=sigma, size=x.shape)
    return torch.tensor(new, dtype=x.dtype, device=x.device)


def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,x.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((x.shape[0],1)), scalingFactor)
    new = x*myNoise
    return torch.tensor(new, dtype=x.dtype, device=x.device)


def permutation(x, max_segments=5, seg_mode="random"):
    x_new = np.zeros(x.shape)
    nPerm = np.random.randint(1, max_segments)
    idx = np.random.permutation(nPerm)

    segs = np.zeros(nPerm+1, dtype=int)
    segs[1:-1] = np.sort(np.random.randint(0, x.shape[0], nPerm-1))
    segs[-1] = x.shape[0]

    pp = 0
    for ii in range(nPerm):
        x_temp = x[segs[idx[ii]]:segs[idx[ii]+1],:]
        x_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return torch.tensor(x_new, dtype=x.dtype, device=x.device)

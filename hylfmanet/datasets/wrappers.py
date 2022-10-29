import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
# from utils import to_pixel_samples
from utils import *


@register('rlfm-vcdnet')
class RLFMVCDnet(object):
    def __init__(self, dataset, randomSeed=None, inp_size = None, augment=False, volume_depth = None, \
        sample_q = None, noise = None, normalize_mode='percentile'):

        self.dataset = dataset
        self.randomSeed = randomSeed

        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q
        # self.shift = get_shift()
        self.normalize_fn = normalize_percentile if normalize_mode == 'percentile' else normalize
        self.noise = noise

        if randomSeed is not None:
            torch.manual_seed(randomSeed)
            



    def __getitem__(self, idx):

        lfstack, volume = self.dataset[idx]
        if self.noise is not None:
            lfstack = (lfstack - 113) / 10
            lfstack = lfstack + 113 + torch.randn(lfstack.shape)*5

        lfstack = self.normalize_fn(lfstack,0.,99.99).clamp(0,1)
        volume = self.normalize_fn(volume,5,99.99).clamp(0,1)
        
        scale = volume.shape[1] / lfstack.shape[1]

        if self.inp_size is not None:
            h,w = self.inp_size, self.inp_size
            H,W = round(h*scale), round(h*scale)
            h0 = random.randint(0, lfstack.shape[1] - h)
            w0 = random.randint(0, lfstack.shape[2] - w)

            H0,W0 = round(h0*scale), round(w0*scale)
            lfstack = lfstack[:,h0:h0+h,w0:w0+w]
            volume = volume[:,H0:H0+H,W0:W0+W]

        

        scale = volume.shape[-1] / lfstack.shape[-1]


        return {
            'inp': lfstack,
            'scale': torch.tensor([scale],dtype=torch.float32),
            'gt': volume,
        }
        # inp 1*Hin*Win
        # coord (Hout*Wout)*2
        # cell (Hout*Wout)*2
        # gt (Hout*Wout)*1

    def __len__(self):
        return len(self.dataset)

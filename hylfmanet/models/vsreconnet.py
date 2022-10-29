# Implementation of HYLFM-A-Net
# modified by kimchange

from argparse import Namespace
import torch
import torch.nn as nn
from torch.nn.modules import pixelshuffle
import torch.nn.functional as F
from models import register


class attenconv(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, 3, 1, 1), nn.ReLU(),
        )

        self.get_atten = nn.Sequential(
            nn.Conv2d(outChannels, outChannels, 1, 1, 0), nn.ReLU(),
            nn.Conv2d(outChannels, outChannels, 1, 1, 0), nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv(x)
        atten = F.adaptive_avg_pool2d(feat, [1,1])
        atten = self.get_atten(atten)
        return feat*atten

class VSRECONNET(nn.Module):
    def __init__(self, args):
        super().__init__()
        inChannels = args.inChannels
        outChannels = args.outChannels
        kSize = args.kSize

        self.convs2d = nn.Sequential(*[
            attenconv(inChannels, 64),
            attenconv(64, 64),
            attenconv(64, 64),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.PixelShuffle(2),
            attenconv(64, 4*(outChannels)),
            attenconv(4*(outChannels), 8*(outChannels)),
        ])
        self.c2z = lambda ipt, c_in_3d=8: ipt.view(ipt.shape[0], c_in_3d, (outChannels), *ipt.shape[2:])
        self.convs3d = nn.Sequential(*[
            nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1), 
            nn.Sigmoid(),
        ])

    def forward(self,x):
        x = self.convs2d(x)
        x = self.c2z(x)
        x = self.convs3d(x)

        return x.squeeze(1)



@register('vsreconnet')
def make_vsreconnet(inChannels, outChannels, kSize=3):
    args = Namespace()

    args.inChannels = inChannels
    args.outChannels = outChannels
    args.kSize = kSize

    return VSRECONNET(args)
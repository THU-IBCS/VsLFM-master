import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register


@register('rlfm')
class RLFM(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.encoder_spec = encoder_spec

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 27
            imnet_in_dim += 3 # attach coord
            if self.cell_decode:
                imnet_in_dim += 3
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp, scale):
        self.feat = self.encoder(inp)
        # self.feat = F.interpolate(self.feat, size = (round((inp.shape[2])*scale.item()), round((inp.shape[3])*scale.item())), mode='bicubic',align_corners=False)
        if len(self.feat.shape) == 4:
            self.feat = F.interpolate(self.feat, size = (round((inp.shape[3])*scale.item()), round((inp.shape[4])*scale.item())),mode='bicubic',align_corners=False).unsqueeze(1) \
                if len(inp.shape) == 5 else F.interpolate(self.feat, size = (round((inp.shape[2])*scale.item()), round((inp.shape[3])*scale.item())),mode='bicubic',align_corners=False).unsqueeze(1)
        else:
            self.feat = F.interpolate(self.feat, size=(inp.shape[1],round((inp.shape[2])*scale.item()), round((inp.shape[3])*scale.item())) ,mode='trilinear',align_corners=False)


        return self.feat


    def forward(self, inp, scale):
        self.gen_feat(inp,scale)
        return self.feat

import argparse
import os
from PIL import Image

import torch
from torchvision import transforms
import sys
sys.path.append('./hylfmanet/')
import models
from utils import *

import numpy
from tifffile import imwrite
import numpy as np
import imageio
from math import ceil

def get_all_abs_path(source_dir):
    path_list = []
    for fpathe, dirs, fs in os.walk(source_dir):
        for f in fs:
            p = os.path.join(fpathe, f)
            path_list.append(p)
    return path_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savefolder', default="./Data/SR_hylfmanetrecon/")
    parser.add_argument('--model', default='./Models/_train_vsreconnet_x3_12119cell-561-20211219-6_20221022/epoch-last.pth') # hylfmanet model to use
    parser.add_argument('--resolution', default='101,1989,1989')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--inp_size', default=237)
    parser.add_argument('--overlap', default=15)
    # if your machine GPU memory is less than 24GB, then you can run
    # python rlfmdemo.py --inp_size 51 --overlap 15
    # to do reconstruction

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    files = get_all_abs_path(args.savefolder[0:args.savefolder.rfind('_')])
    if not os.path.exists(args.savefolder):
        os.mkdir(args.savefolder)
    for file in files:
        print(file)
        inputfile = file
        lfstack = torch.tensor(np.array(imageio.volread(inputfile),dtype=np.float32))

        lfstack_low  = np.percentile(lfstack, 0.2) * 0
        lfstack_high = np.percentile(lfstack, 99.99)
        eps = 1e-3

        lfstack = (lfstack - lfstack_low) / (lfstack_high - lfstack_low + eps)


        model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
        d, h, w = list(map(int, args.resolution.split(',')))


        scale = h / lfstack.shape[1]

        inp_all = lfstack.unsqueeze(0)

        inp_size = int(args.inp_size)
        inp_size = min(inp_size, inp_all.shape[-1])
        ret = torch.zeros([d,h,w])

        # overlap sigmoid weighted
        overlap = int(args.overlap)
        overlapVolume = round(overlap * scale)
        if overlap:
            edge = torch.sigmoid((torch.arange(overlapVolume) - overlapVolume//2 ) / overlapVolume *15)
            weight = torch.cat([edge, edge.max()*torch.ones(round(inp_size*scale) - 2*len(edge)),edge.flip(0)],dim=0).view(-1,1) @ \
                torch.cat([edge, edge.max()*torch.ones(round(inp_size*scale) - 2*len(edge)),edge.flip(0)],dim=0).view(1,-1) + 1e-3
            weight = weight.unsqueeze(0) # 1,h,w
            del edge
        else:
            weight = torch.ones(round(inp_size*scale), round(inp_size*scale))
            weight = weight.unsqueeze(0) # 1,h,w
        base = torch.zeros_like(ret)

        t0 = time.time()

        for h0 in [i*(inp_size-overlap) for i in range(1+ceil((inp_all.shape[-2]-inp_size)/(inp_size-overlap)))]:
            for w0 in [i*(inp_size-overlap) for i in range(1+ceil((inp_all.shape[-1]-inp_size)/(inp_size-overlap)))]:
                inp = inp_all[:,:,h0:h0+inp_size,w0:w0+inp_size]

                with torch.no_grad():
                    pred = model( ((inp - 0) / 1).cuda(), torch.tensor([scale],dtype=torch.float32).unsqueeze(0))

                pred[pred<0] = 0
                pred[torch.isnan(pred)] = 0
                pred[torch.isinf(pred)] = 0
                    
                pred = pred.cpu()

                ret[:,round(h0*scale):round((h0+inp_size)*scale),round(w0*scale):round((w0+inp_size)*scale)]= \
                    ret[:,round(h0*scale):round((h0+inp_size)*scale),round(w0*scale):round((w0+inp_size)*scale)] + pred * weight[:,0:pred.shape[-2],0:pred.shape[-1] ]
                base[:,round(h0*scale):round((h0+inp_size)*scale),round(w0*scale):round((w0+inp_size)*scale)]= \
                    base[:,round(h0*scale):round((h0+inp_size)*scale),round(w0*scale):round((w0+inp_size)*scale)] + weight[:,0:pred.shape[-2],0:pred.shape[-1] ]
                pred = None
        ret = ret / (base)
        print(time.time()-t0)
        # imwrite(args.savefolder + inputfile[-inputfile[-1::-1].find('/'):][0:-4]+'_hylfmanet.tif',np.uint16(ret[30:61,960:960+751, 318:318+1241] * 2000), imagej=True, metadata={'axes': 'ZYX'}, compression ='zlib')
        imwrite(args.savefolder + inputfile[-inputfile[-1::-1].find('/'):][0:-4]+'_hylfmanet.tif',np.uint16(ret[:,:, :] * 2000), imagej=True, metadata={'axes': 'ZYX'}, compression ='zlib')
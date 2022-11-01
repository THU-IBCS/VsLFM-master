"""Testing script for LF-image SR.
Train a model:
        python test.py
See Readme.md for more testing details.
"""

import torch
from Models.vsnet import VsNet
from torch.autograd import Variable
import argparse
import numpy as np
import math
import os
import tifffile as tiff

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # Use 2 GPUs[0,1]
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Use 1 GPU[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def main(cfg):
    '''
    The input: LR data 'LR.tif' , size of [169,153,153]
    The output: SR data 'SR.tif' , size of [169,459,459]
    Note that, the input data is cropped into 9 small with the patch size of [169,69,69] before being fed into the network,
    and the 9 output SR patches would be spliced into one SR image finally.
    '''

    dir_images_tif = './Data/LR/mito_LR.tif'  # File path of the LR image for test
    dir_save_path = './Data/SR/'            # File path to save of the SR image
    name = 'mito_SR.tif'                    # File name to save of the SR image


    folder = os.path.exists(dir_save_path)
    if not folder:
        os.makedirs(dir_save_path)

    # network
    net = VsNet(angRes=13, K=4, n_block=4, channels=64, upscale_factor=3).to(cfg.device)
    # net = torch.nn.DataParallel(net, device_ids=[0, 1])   # Use 2 GPUs[0,1]
    net = torch.nn.DataParallel(net, device_ids=[0])        # Use 1 GPU[0]


    # Load the pretrained model for test
    dir_pth_name = './Models/pretrained_models/our_model.pth.tar'  # File path of the pretrained model
    pth_file_path = [dir_pth_name]
    model = torch.load(pth_file_path[0])
    net.load_state_dict(model['state_dict'])

    # Set patch parameters   | 3*patchsize - 2*(overlap//3) = image_size[2] | stride = patchsize-overlap//3a
    patchsize = 69                   # The cropped LR patches size
    overlap = 27*3                           # the overlapping region of cropped the SR patches, 3 is the upscale factor
    stride = patchsize-overlap//3         # the stride of cropped LR patches, 3 is the upscale factor

    data_LR = tiff.imread(dir_images_tif)  # Size of [169,153,153]


    data_LR = torch.from_numpy(data_LR.astype(np.float32))
    C, H, W = data_LR.shape[0], data_LR.shape[1], data_LR.shape[2]

    # Crop LR data into 9 patches
    data_patches = []
    for h in range(0, H - patchsize + 1, stride):
        for w in range(0, W - patchsize + 1, stride):
            tempLR = data_LR[:, h: h + patchsize, w: w + patchsize]
            data_patches.append(tempLR)
    data_patches = torch.stack(data_patches, 0)                    # Size of [9,169,69,69]
    n = data_patches.shape[0]

    max_value =data_patches.max()

    output=[]
    for file_num in range(n):
        data = data_patches[file_num,:, :, :]/max_value            # Normalization
        data = data.unsqueeze(0)
        data = Variable(data).to(cfg.device)

        with torch.no_grad():
            out = net(data)
            out = torch.clamp(out,0,1)                             # Limit the value to the range of [0,1]

        torch.cuda.empty_cache()                                   # Release GPU memory
        output.append(out.squeeze())
    output = torch.stack(output, 0) * max_value                    # Size of [n,169,h,w]
    output = output.cpu().numpy()   
    out_cat = patch_cat(output, overlap)                           # Merge 9 output SR patches into one SR image
    
    tiff.imsave(dir_save_path + name, out_cat.astype(np.uint16))   # Save the output SR image



def patch_cat(patch, overlap):
    # Reture the merged SR image from n small patches

    Nbx = int(math.sqrt(patch.shape[0]))  # Number of patches along x axis
    Nby = int(math.sqrt(patch.shape[0]))  # Number of patches along y axis
    stride = patch.shape[2] - overlap     # Stride = 207 - 81

    a = - 0.5
    x = (np.arange(1, overlap + 1))
    y = 1 / (1 + np.exp(a * (x - 14)))    # Sigmoid function for seamless stitching with overlapping images

    z = np.concatenate((y, max(y) * np.ones((patch.shape[2] - 2 * overlap)), y[::-1])) + 0.001
    z = z.reshape((-1, 1))
    W = z.transpose(1, 0) * z
    W = np.expand_dims(W, 0).repeat(patch.shape[1], axis=0)

    img = np.zeros(
        (patch.shape[1], patch.shape[2] * Nbx - overlap * (Nbx - 1), patch.shape[3] * Nby - overlap * (Nbx - 1)))
    W_f = np.zeros(
        (patch.shape[1], patch.shape[2] * Nbx - overlap * (Nbx - 1), patch.shape[3] * Nby - overlap * (Nbx - 1)))
    
    # Image stitching
    for u in range(Nbx):
        for v in range(Nby):
            x_begin = u * stride
            x_end = x_begin + patch.shape[2] - 1
            y_begin = v * stride
            y_end = y_begin + patch.shape[3] - 1
            img[:, x_begin: x_end + 1, y_begin:y_end + 1] = img[:, x_begin: x_end + 1, y_begin:y_end + 1] + W * patch[u * Nbx + v,:,:, :]
            W_f[:, x_begin: x_end + 1, y_begin:y_end + 1] = W_f[:, x_begin: x_end + 1, y_begin:y_end + 1] + W
    img = img / W_f
    return img


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

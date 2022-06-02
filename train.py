"""Training script for LF-image SR.
Train a model:
        python train.py
See Readme.md for more training details.
"""

import torch
import logging
import argparse
from Models.vsnet import VsNet
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils.utils import *
from math import log10
import os
from pathlib import Path

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use 2 GPUs[0,1]
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use 1 GPU[0]

# Create the log path for saving the models and log.txt
log_path = './Models/model/'
folder = os.path.exists(log_path)
if not folder:
    os.mkdir(log_path)
file_log_name = log_path +'/log.txt'
logger = get_logger(file_log_name)

# Training settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=13, help="angular resolution")
    parser.add_argument("--K", type=int, default=4, help="number of cascades")
    parser.add_argument("--n_block", type=int, default=4, help="number of Inter-Blocks")
    parser.add_argument("--upscale_factor", type=int, default=3, help="upscale factor")
    parser.add_argument("--channels", type=int, default=64, help="channels")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=40, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=10, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')

    parser.add_argument('--trainset_dir', type=str, default='./Datasets/')
    parser.add_argument('--model_name', type=str, default='LFSRmodel')
    parser.add_argument('--load_pretrain', type=bool, default=False) # if you want to load the pretrained models, set to True, and set the model_path
    parser.add_argument('--model_path', type=str, default='./Models/pretrained-models/model.pth.tar')

    return parser.parse_args()


def train(train_loader, cfg):
    logger.info('start training!')
    logger.info('batch_size:{:3d}\t learning rate={:.6f}\t  n_steps={:3d}'.format(cfg.batch_size, float(cfg.lr),
                                                                                  cfg.n_steps))
    # Create the model given opt.model and other options
    net = VsNet(angRes=cfg.angRes, K=cfg.K, n_block=cfg.n_block, channels=cfg.channels, upscale_factor=cfg.upscale_factor).to(cfg.device)
    net.apply(weights_init_xavier)
    cudnn.benchmark = True

    # If the training process ia interrupted, you can continue the training by loading the last pretrained weights
    # and set the start epoch from the initial 0 to the cfg.start_epoch(the number of the last epoch before interrupted.
    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            pth_file_path = [cfg.model_path]
            model = torch.load(pth_file_path[0])  # load the pretrained model
            net.load_state_dict({k.replace('module.', ''): v for k, v in model['state_dict'].items()})
            cfg.start_epoch = model["epoch"]
        else:
            print("=> no model found at '{}'".format(cfg.load_model))

    # net = torch.nn.DataParallel(net, device_ids=[0, 1])  # use 2 GPUs[0,1]
    net = torch.nn.DataParallel(net, device_ids=[0])  # use GPU[0]
    criterion_Loss = torch.nn.L1Loss().to(cfg.device)   # L1 loss function
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)  # optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    psnr_epoch = []
    loss_epoch = []
    loss_list = []
    psnr_list = []

    for idx_epoch in range(cfg.n_epochs):
        for idx_iter, (data, label) in enumerate(train_loader):
            data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)  # data is the LR, label is the HR(groundtruth)

            out = net(data)   # Put data into the model network

            loss = criterion_Loss(out, label)  # Calculate the training loss
            optimizer.zero_grad()    # Optimizer
            loss.backward()
            optimizer.step()
            psnr_epoch.append(cal_psnr(out.data.cpu(), label.data.cpu()))
            loss_epoch.append(loss.data.cpu())
        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean())) # calculate the training loss
            psnr_list.append(float(np.array(psnr_epoch).mean())) # calculate the training PSNR
            # Print training losses, PSNR and save logging information
            logger.info(
                'Epoch:{:3d}\t loss={:.6f}\t  PSNR={:.5f}'.format(idx_epoch + 1, float(np.array(loss_epoch).mean()),
                                                                  float(np.array(psnr_epoch).mean())))
            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'loss': loss_list,
                'psnr': psnr_list,
            }, save_path= log_path,
                filename=str(cfg.model_name) + '_'+ str(cfg.angRes) + 'x' + str(cfg.angRes) + '_' +
                         str(cfg.upscale_factor) + 'xSR_epoch' + str(idx_epoch + 1) + '.pth.tar')
            psnr_epoch = []
            loss_epoch = []
        scheduler.step()
    logger.info('finish training!')

# Calculate the PSNR for check manually
def cal_psnr(img1, img2):
    _, _, h, w = img1.size()
    mse = torch.sum((img1 - img2) ** 2) / img1.numel()
    psnr = 10 * log10(1 / mse)
    return psnr

# Save the trained model checkpoints
def save_ckpt(state, save_path=log_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))

# Weight initial
def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)


def main(cfg):
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir)
    train_loader = DataLoader(dataset=train_set, num_workers=6, batch_size=cfg.batch_size, shuffle=True)
    train(train_loader, cfg)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

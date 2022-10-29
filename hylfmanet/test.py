import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

# for tensorboard visulize (just a reference)
def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, writer = None, config=None, EPOCH=0, tensorboard_image_writing = True):
    model.eval()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()


        if eval_bsize is None:
            with torch.no_grad():
                pred = model(batch['inp'], batch['scale'])

        pred = pred.squeeze(1) if pred.shape[1] == 1 else pred

        if (len(pred.shape) == 5) and (pred.shape[1] != 1):
            pred = pred[:,1,:,:,:]
        
        pred = pred.clamp_(0, 1)
        pred[torch.isnan(pred)] = 0
        pred[torch.isinf(pred)] = 0

        if 'writer' in locals().keys() and tensorboard_image_writing == True:
            tensorboard_image_writing = False

            writer.add_image('val_input', utils.cmap[ (batch['inp'][0,84,:,:]*255).long()].cpu() ,dataformats='HWC',global_step=EPOCH)          
            writer.add_image('val_pred', utils.cmap[ (      torch.cat([pred[0,:,:,:].max(0).values,pred[0,:,:,:].max(2).values.permute(1,0)],dim=1)      *255).long()].cpu() ,dataformats='HWC',global_step=EPOCH)
            writer.add_image('val_gt', utils.cmap[ (        torch.cat([batch['gt'][0,:,:,:].max(0).values,batch['gt'][0,:,:,:].max(2).values.permute(1,0)],dim=1)     *255).long()].cpu() ,dataformats='HWC',global_step=EPOCH)

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), batch['inp'].shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()

# to do: 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)
    print('result: {:.4f}'.format(res))

## HyLFM-A-Net

For the problem of low computational efficiency in the second step of VsLFM, we developed a new end-to-end reconstruction network named HyLFM-A-Net, extended from the existing HyLFM-Net to show that our VsLFM is compatible with different reconstruction algorithms. Channel attention mechanism is embeded into the existing HyLFM-Net to replace iterative tomography for two orders of magnitude reduction in computational costs. But it should be noted that the robustness to sample aberrations will reduce relative to using iterative tomography, since the end-to-end network does not consider the influence of optical aberrations.

### Preparation

The HyLFM-A-Net is used to accelerate reconstruction through full-supervised network, whereby paired dataset is required for training. Users can obtain training data pairs using iterative tomography on the Vs-Net output. 

Some extra python packages are required:
```bash
tensorboardX
pyyaml
imageio
tqdm
```

The GPU memory of 24 GB is required, or some image crop operations should be considered.

### How to train

We use yaml to manage different training options. Users should modify the 'yaml' in configs, and set the filepaths of training pairs. Then run:
```bash
python train_rlfm.py --config ./configs/train-rlfm/train_vsreconnet_x3_mito_demo.yaml --tag 20221029 --gpu 0
```
The code will automatically save the model parameters in '../save'. A pre-trained model can be found in 'VsLFM-master/Models/_train_vsreconnet_x3_12119cell-488-20211219-6_20221029/epoch-last.pth'.

### How to track the training process
```bash
tensorboard --bind_all --logdir=../save --samples_per_plugin "images=1000"
```

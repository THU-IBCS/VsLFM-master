## HyLFM-A-Net

This subdir shows an end to end reconstruction alternative, which is inspired by HyLFM-Net.

### Preparation

The HyLFM-A-Net is used for accelerate reconstruction, paired dataset is required. Here, you can get paired data using iterative tomography, then reconstruct for example 10 paired data. 

Some extra python packages are required:
```bash
tensorboardX
pyyaml
imageio
tqdm
```

The GPU memory better > 20GB, or some crop operations will be considered.

### How to train

We use yaml to manage different training options. So, you need to modify the 'yaml' in configs, tell the code where is the input training lf, where is the paired volume. Then run:

```bash
python train_rlfm.py --config ./configs/train-rlfm/train_vsreconnet_x3_mito_demo.yaml --tag 20221029 --gpu 0
```
The code will automatically save the model parameters in '../save'. An example model trained can be found at 'VsLFM-master/Models/_train_vsreconnet_x3_12119cell-561-20211219-6_20221022/epoch-last.pth'

### How to tract training process
```bash
tensorboard --bind_all --logdir=../save --samples_per_plugin "images=1000"
```

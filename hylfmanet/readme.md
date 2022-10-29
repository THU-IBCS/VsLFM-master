## HyLFM-A-Net

This subdir shows an end to end reconstruction alternative, which is inspired by HyLFM-Net.

### How to train

We use yaml to manage different training options. So, firstly you should modify the yaml in configs. Then run:
```bash
python train_rlfm.py --config ./configs/train-rlfm/train_vsreconnet_x3_12119cell-561-20211219-6.yaml --tag 20221029 --gpu 0
```

### How to tract training process
```bash
tensorboard --bind_all --logdir=../save --samples_per_plugin "images=1000"
```

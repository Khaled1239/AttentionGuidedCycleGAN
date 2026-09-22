# Attention-Guided CycleGAN — CNMC to ALL-IDB

PyTorch implementation of the main attention-guided CycleGAN mechanisms
described in the supplied paper.

Source: CNMC
Target: ALL-IDB

Features:
- Spatial attention: average/max pooling -> 7x7 convolution -> sigmoid
- Attention-guided foreground translation + background retention
- ResNet generator
- 70x70 PatchGAN discriminator
- Least-squares GAN loss
- Cycle-consistency L1 loss
- Pixel L1 loss
- 50-image replay buffer
- 200-epoch default schedule
- CUDA support
- Checkpoints, sample images, loss CSV and loss plots

## Dataset structure

data/
  CNMC/
    ALL/
    Normal/
  ALL_IDB/
    ALL/
    Normal/

The GAN ignores class labels during translation training.

## Install

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

Check GPU:
python check_environment.py

Train:
python train.py --epochs 200 --batch_size 4

Translate CNMC -> ALL-IDB:
python translate.py --checkpoint checkpoints/latest.pt

Plot losses:
python plot_losses.py

## Important
Keep the held-out ALL-IDB test set outside the training folders if you want
to preserve the paper's target-test protocol.

This is a research implementation based on the paper's described method,
not an official source-code release.

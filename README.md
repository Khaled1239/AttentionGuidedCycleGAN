# Attention-Guided CycleGAN — C-NMC to ALL-IDB

PyTorch implementation of an **Attention-Guided CycleGAN** for medical image domain adaptation between **C-NMC** and **ALL-IDB** leukemia cell image datasets.

The project implements the main mechanisms described in the reference paper, including spatial attention, attention-guided image translation, CycleGAN losses, and PatchGAN discrimination.

> **Note:** This is an independent research implementation based on the method described in the reference paper. It is **not the official source code** of the paper.

---

## Project Overview

The main objective is to reduce the visual/domain differences between:

* **Source domain:** C-NMC
* **Target domain:** ALL-IDB

The translation process converts C-NMC images into an ALL-IDB-like visual domain while attempting to preserve relevant cell structures.

The attention mechanism helps the model focus the translation on relevant image regions while retaining other parts of the original image.

### Main Pipeline

```text
C-NMC Image
     │
     ▼
┌──────────────────────┐
│   Generator G_ST     │
│                      │
│ Encoder              │
│ Spatial Attention    │
│ Residual Blocks      │
│ Decoder              │
└──────────┬───────────┘
           │
           ▼
   Attention-Guided
   Translated Image
           │
           ▼
      ALL-IDB Style
           │
           ▼
    PatchGAN D_T
           │
           ▼
   Domain Evaluation
```

---

## Main Components

### 1. Spatial Attention

The attention module follows the general structure:

```text
Feature Map
     │
     ├── Average Pooling
     │
     └── Max Pooling
             │
             ▼
        Concatenation
             │
             ▼
         7×7 Conv
             │
             ▼
          Sigmoid
             │
             ▼
      Spatial Attention
```

The resulting attention map is used to guide the translation process.

---

### 2. Attention-Guided Generator

The generator consists of:

* Initial convolution layers
* Downsampling layers
* Spatial attention module
* 6 residual blocks
* Upsampling layers
* Final image reconstruction layer

The attention-guided output is calculated as:

```text
output =
    attention × generated_image
    +
    (1 - attention) × input_image
```

This allows the model to modify relevant regions while retaining other regions of the original image.

---

### 3. 70×70 PatchGAN Discriminator

The discriminator follows a **70×70 PatchGAN** architecture.

It receives an image and produces a spatial score map rather than a single classification value.

For a `128 × 128` input, the discriminator produces:

```text
[Batch, 1, 14, 14]
```

The discriminator uses raw outputs because the project uses **Least Squares GAN (LSGAN)** loss.

---

### 4. Training Losses

The training process uses:

#### GAN Loss

Encourages translated images to resemble the target domain.

#### Cycle Consistency Loss

Encourages the translation process to preserve the original image content.

```text
Source → Target → Source
```

#### Pixel Loss

Penalizes excessive changes between the original image and the attention-guided translated image.

### Total Loss

```text
L_total =
    λ_gan L_gan
    +
    λ_cycle L_cycle
    +
    λ_pixel L_pixel
```

Current values:

```text
λ_gan    = 0.5
λ_cycle  = 10
λ_pixel  = 1
```

---

# Project Structure

```text
AttentionGuided_CycleGAN/
│
├── Models/
│   ├── Attention.py
│   ├── AttentionGuidedCyclegan.py
│   ├── Discriminator.py
│   ├── Generator.py
│   └── __init__.py
│
├── ModelsBackUp/
│   ├── Attention.py
│   ├── AttentionGuidedCyclegan.py
│   ├── Discriminator.py
│   └── Generator.py
│
├── Utils/
│   ├── Dataset.py
│   ├── ImageUtils.py
│   ├── Losses.py
│   ├── ReplayBuffer.py
│   └── __init__.py
│
├── Data/
│   ├── C_NMC/
│   │   ├── ALL/
│   │   └── Normal/
│   │
│   └── ALL_IDB/
│       ├── ALL/
│       └── Normal/
│
├── Checkpoints/
│
├── Outputs/
│   ├── DomainEval/
│   ├── Plots/
│   └── Samples/
│
├── CheckEnv.py
├── EvaluateDiscriminator.py
├── PlotLosses.py
├── Train.py
├── TrainBackUp.py
├── Translate.py
├── requirements.txt
└── README.md
```

---

# Dataset

The expected dataset structure is:

```text
Data/
├── C_NMC/
│   ├── ALL/
│   └── Normal/
│
└── ALL_IDB/
    ├── ALL/
    └── Normal/
```

The class labels are used to organize the images but are **not used as supervised class labels during CycleGAN translation training**.

The model performs unpaired image-to-image domain translation between the two datasets.

> Keep the held-out ALL-IDB test images outside the training folders if you want to preserve the target-test protocol described in the reference paper.

---

# Installation

## 1. Clone the repository

```powershell
git clone https://github.com/Khaled1239/AttentionGuidedCycleGAN.git
cd AttentionGuidedCycleGAN
```

## 2. Create a virtual environment

The project was developed using **Python 3.10**.

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

# Running the Project

The recommended workflow is:

```text
1. Check Environment
        ↓
2. Prepare Dataset
        ↓
3. Train CycleGAN
        ↓
4. Plot Training Losses
        ↓
5. Translate C-NMC → ALL-IDB
        ↓
6. Evaluate Discriminator
```

---

# Step 1 — Check Environment

Before training, verify Python, PyTorch, and CUDA:

```powershell
python CheckEnv.py
```

The script checks whether the environment can access the NVIDIA GPU.

For CUDA-enabled PyTorch, the expected result includes:

```text
CUDA available: True
```

---

# Step 2 — Prepare Dataset

Place the datasets inside:

```text
Data/
```

with the following structure:

```text
Data/
├── C_NMC/
│   ├── ALL/
│   └── Normal/
│
└── ALL_IDB/
    ├── ALL/
    └── Normal/
```

Images are resized during preprocessing according to the current training configuration.

---

# Step 3 — Train the Model

Start training with:

```powershell
python Train.py
```

### Current experimental configuration

The current experiment uses a reduced configuration because of limited computational resources:

```text
Image size    : 64 × 64
Epochs        : 30
Batch size    : 1
Learning rate : 0.0001
β1            : 0.5
β2            : 0.999

λ_gan         : 0.5
λ_cycle       : 10
λ_pixel       : 1

Replay buffer : 50 images
```

For an RTX 2050 4 GB GPU, a small batch size is recommended:

```powershell
python Train.py --batch_size 1
```

If sufficient GPU memory is available:

```powershell
python Train.py --batch_size 2
```

> The current 64×64 resolution and 30-epoch schedule are used as a resource-limited experimental configuration. They are smaller than the settings described in the reference paper.

---

## Training Outputs

Checkpoints are saved to:

```text
Checkpoints/
```

Example:

```text
Checkpoints/
├── Epoch_010.pt
├── Epoch_020.pt
├── Epoch_030.pt
└── Latest.pt
```

Training samples are saved to:

```text
Outputs/Samples/
```

Training history is saved to:

```text
Outputs/TrainingLog.csv
```

---

# Step 4 — Plot Training Losses

After or during training, visualize the recorded losses using:

```powershell
python PlotLosses.py
```

The plots are saved to:

```text
Outputs/Plots/
```

Examples:

```text
GANLoss.png
GLoss.png
DLoss.png
cycleLoss.png
pixelLoss.png
```

These plots can be used to inspect generator, discriminator, cycle-consistency, pixel, and GAN loss behavior.

---

# Step 5 — Translate C-NMC → ALL-IDB

After training, use the trained generator to translate C-NMC images:

```powershell
python Translate.py
```

If the script supports specifying a checkpoint:

```powershell
python Translate.py --checkpoint Checkpoints/Latest.pt
```

The translation process is:

```text
C-NMC
  │
  ▼
G_ST
  │
  ├── Spatial Attention
  │
  ▼
Attention-Guided Translation
  │
  ▼
ALL-IDB-like Image
```

The translated images are saved in the configured output directory.

---

# Step 6 — Evaluate the Discriminator

After generating translated images, evaluate the target-domain discriminator:

```powershell
python EvaluateDiscriminator.py
```

The evaluation compares discriminator responses between:

```text
Real ALL-IDB
       vs
Translated C-NMC
```

Results are saved to:

```text
Outputs/DomainEval/
```

including:

```text
discriminator_histogram.png
discriminator_scores.png
discriminator_statistics.csv
```

### Important

The discriminator produces **PatchGAN / LSGAN scores**, not calibrated probabilities.

Therefore:

```text
0.5 ≠ 50% probability
```

Instead, examine the score distributions and statistics of real ALL-IDB and translated C-NMC images.

Similar discriminator responses can indicate that translated images are being treated similarly to target-domain images. However, discriminator score similarity alone is not sufficient to establish complete domain alignment.

---

# Training Configuration

The current experiment uses the following configuration:

| Parameter                 |  Current Value |
| ------------------------- | -------------: |
| Image size                |        64 × 64 |
| Epochs                    |             30 |
| Batch size                |              1 |
| Learning rate             |           1e-4 |
| Adam β1                   |            0.5 |
| Adam β2                   |          0.999 |
| λ GAN                     |            0.5 |
| λ Cycle                   |             10 |
| λ Pixel                   |              1 |
| Replay buffer             |             50 |
| Generator residual blocks |              6 |
| Discriminator             | 70×70 PatchGAN |

The reduced image resolution and training duration are intentional limitations of the current experiment due to available computational resources.

The reference paper uses a larger training configuration. Therefore, the current results should be interpreted as a resource-limited experimental implementation rather than a direct reproduction of the paper's full training setup.

---

# Research Workflow

```text
┌──────────────────────┐
│ C-NMC + ALL-IDB      │
│ Dataset Preparation  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Attention-Guided     │
│ CycleGAN Training    │
└──────────┬───────────┘
           │
           ├───────────────┐
           ▼               ▼
┌─────────────────┐ ┌─────────────────┐
│ Training Loss   │ │ Generated       │
│ Analysis        │ │ Images          │
└─────────────────┘ └────────┬────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Discriminator    │
                    │ Evaluation       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Domain Alignment │
                    │ Analysis         │
                    └──────────────────┘
```

---

# Important Notes

### Independent implementation

This project is an independent implementation based on the architecture and methodology described in the reference paper.

It is **not the official source code** released by the paper's authors.

Some implementation details may therefore differ from the original implementation.

### Checkpoints

Large PyTorch checkpoint files should generally not be committed to Git:

```text
*.pt
*.pth
```

Keep them locally under:

```text
Checkpoints/
```

and add the directory to `.gitignore`.

### Dataset

The original datasets should not be included directly in this repository unless their respective licenses and redistribution terms permit it.

### Experimental outputs

Generated images, checkpoints, plots, and evaluation results can be kept locally during experimentation.

For a clean GitHub repository, large datasets, model checkpoints, generated outputs, and Python cache files should be excluded from version control.

---

# Reference

This implementation is based on the methodology described in the supplied research paper concerning attention-guided CycleGAN for reducing domain differences between C-NMC and ALL-IDB leukemia cell images.

The implementation focuses on:

* Spatial attention
* Attention-guided image translation
* Cycle consistency
* Pixel consistency
* LSGAN
* PatchGAN discrimination
* C-NMC → ALL-IDB domain adaptation

---

# Author

**ANTELIERE**

Research / Data Science Project

PyTorch implementation for medical image domain adaptation.

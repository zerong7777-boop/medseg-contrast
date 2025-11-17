# medseg-contrast

> Reproducible medical image segmentation baselines & contrast experiments.

![Language](https://img.shields.io/github/languages/top/zerong7777-boop/medseg-contrast)
![License](https://img.shields.io/github/license/zerong7777-boop/medseg-contrast)

This repository provides a small, self-contained framework for **2D medical image segmentation**, with multiple **CNN-based baselines** and initial **contrastive-learning experiments**.  
The current code focuses on the **ISIC 2016 skin lesion segmentation** benchmark and is designed to be easy to run, modify, and extend.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
  - [Clone](#clone)
  - [Environment](#environment)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
  - [Quick Start (ISIC16)](#quick-start-isic16)
  - [Custom Runs](#custom-runs)
- [Utilities](#utilities)
- [Extending the Codebase](#extending-the-codebase)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

`medseg-contrast` aims to:

- Provide **reproducible baselines** for medical image segmentation (currently 2D, ISIC16).
- Compare several **UNet-like and attention-based architectures** under a unified training pipeline.
- Offer a simple playground for **contrastive learning ideas** on medical segmentation (e.g., contrastive feature losses in `loss.py`).
- Serve as a lightweight starting point for your own segmentation or representation learning experiments.

---

## Project Structure

At the top level:

```text
medseg-contrast/
├── ResNet/              # ResNet backbone implementations
├── UKAN/                # UKAN model variants (segmentation)
├── UTNet/               # UTNet segmentation networks
├── pytorch_dcsaunet/    # DCSA-UNet implementation
├── utils/               # Training, logging, and metric utilities
├── Dataloader_ISIC16.py # ISIC16 dataset loader
├── isic16_run.py        # Main training / evaluation script for ISIC16
├── ISIC16.sh            # Example shell script to run ISIC16 experiments
├── batch_pred2gray.sh   # Helper script to post-process prediction masks
├── loss.py              # Segmentation and contrastive loss functions
├── .gitignore
├── LICENSE              # Apache-2.0
└── README.md
```

> Note: The current version is centered on ISIC16. You can plug in other datasets by creating a new dataloader and adapting the training script.

---

## Requirements

Recommended:

- **Python** ≥ 3.8
- **PyTorch** + **torchvision** (GPU support strongly recommended)
- Common scientific Python stack:
  - `numpy`
  - `scikit-image` or `Pillow` (image I/O & processing)
  - `opencv-python` (optional, for additional image utilities)
  - `tqdm` (progress bars)
  - `matplotlib` (optional, plotting)

Install them via `pip` or `conda` as you prefer. Example:

```bash
conda create -n medseg-contrast python=3.8
conda activate medseg-contrast

# Example (adjust to your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Common utilities
pip install numpy scikit-image opencv-python tqdm matplotlib
```

---

## Getting Started

### Clone

```bash
git clone https://github.com/zerong7777-boop/medseg-contrast.git
cd medseg-contrast
```

### Environment

Make sure you have activated the environment where PyTorch and the other dependencies are installed:

```bash
conda activate medseg-contrast   # or your own env name
```

---

## Data Preparation

The current code is written for **ISIC 2016** skin lesion segmentation.

1. **Download ISIC 2016**  
   Download the images and masks from the official ISIC 2016 challenge website.

2. **Organize the dataset**  
   A typical layout (adapt to your own paths):

   ```text
   /path/to/ISIC16/
   ├── train/
   │   ├── img/    # training images
   │   └── mask/   # corresponding GT masks
   ├── val/
   │   ├── img/    # validation images
   │   └── mask/
   └── test/
       ├── img/    # test images (no GT or separate GT folder)
       └── mask/   # (optional, if you keep GT here too)
   ```

3. **Point the code to your dataset path**  

   - Either directly edit **`Dataloader_ISIC16.py`** to set the root path, or  
   - Use the arguments / variables exposed in **`isic16_run.py`** or **`ISIC16.sh`** to pass the dataset root.

   > Tip: open `Dataloader_ISIC16.py` and check how it expects the directory structure (e.g., subfolder names like `train`, `val`, `test`, `img`, `mask`).

---

## Training and Evaluation

### Quick Start (ISIC16)

The simplest way to launch a default ISIC16 experiment is to use the provided shell script:

```bash
bash ISIC16.sh
```

What this usually does (depending on how you configure it):

- Sets environment variables (GPU, dataset path, experiment name).
- Calls `python isic16_run.py` with a default configuration (e.g., model, LR, epochs).

Before running:

- Open `ISIC16.sh`.
- Adjust:
  - `DATA_ROOT` or similar variables to your ISIC16 path.
  - GPU / device index if needed.

### Custom Runs

To see all available options:

```bash
python isic16_run.py --help
```

Depending on how you implement arguments in `isic16_run.py`, typical options may include:

- `--dataset_root` or similar: path to ISIC16
- `--model`: e.g. `UNet`, `UTNet`, `DCSAUNet`, `UKAN`
- `--epochs`: number of training epochs
- `--batch_size`: batch size
- `--lr`: learning rate
- `--save_dir`: directory for checkpoints and logs
- Flags or parameters enabling **contrastive loss** or additional regularization (see `loss.py`)

A hypothetical example (adjust to match your actual arguments):

```bash
python isic16_run.py \
  --dataset_root /path/to/ISIC16 \
  --model UNet \
  --epochs 200 \
  --batch_size 8 \
  --lr 1e-3
```

> Please check `isic16_run.py` to confirm the exact argument names and defaults.

---

## Utilities

### `batch_pred2gray.sh`

This script is intended to post-process **prediction masks** after inference, e.g., converting multi-class or binary predictions into 8-bit grayscale PNGs (0/255) for evaluation and visualization.

A typical workflow:

1. Run your model and save raw predictions into a directory, e.g.:

   ```text
   results/ISIC16/UNet/test_pred/
   ```

2. Edit `batch_pred2gray.sh` to:

   - Point `ROOT` to the directory containing model outputs.
   - Adjust model list or structure if necessary.

3. Execute:

   ```bash
   bash batch_pred2gray.sh
   ```

This will generate something like:

```text
results/ISIC16/UNet/test_pred_gray/
```

which can then be used for computing dice/IoU or for visualization.

---

## Extending the Codebase

Some ideas on how to extend this repository:

- **New architectures**  
  Add new segmentation networks under a new folder (e.g., `MyNet/`) and plug them into `isic16_run.py`.

- **New datasets**  
  Implement `Dataloader_MYDATASET.py` following the pattern in `Dataloader_ISIC16.py`, and add an option in `isic16_run.py` to select it.

- **Stronger contrastive learning**  
  Use `loss.py` as a starting point and:
  - Add patch-level or pixel-level contrastive losses.
  - Experiment with global feature projections (e.g., InfoNCE, SupCon).
  - Visualize embeddings (e.g., t-SNE or UMAP) to study representation quality.

- **Logging & visualization**  
  Integrate TensorBoard, Weights & Biases, or your favorite experiment tracker for monitoring training curves and sample outputs.

---

## License

This project is licensed under the **Apache License 2.0**.  
See the [LICENSE](./LICENSE) file for details.

---

## Acknowledgements

This repository bundles or re-implements several backbone and segmentation architectures (ResNet, UNet variants, UTNet, DCSA-UNet, UKAN, etc.).  
Please also respect the original licenses and papers of these models if you use or extend them in your own work.

If you build on this codebase in research, you are encouraged (but not required) to acknowledge the repository in your paper or project.

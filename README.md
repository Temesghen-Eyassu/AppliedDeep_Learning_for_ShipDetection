# Applied_DeepLearning_for_ShipDetection

This repository provides a **PyTorch implementation of U-Net** for ship detection in Sentinel-1 SAR imagery (VV + VH).  
It is organized for development in **PyCharm** and can also be run from the command line.

---

## Contents
- `data/Patches/`
  - `images/` – SAR input patches
  - `masks/` – Ground truth ship masks
- `src/` – Model, dataset, trainer, and utilities
- `configs/` – Default configuration (paths, training settings)
- `checkpoints/` – Saved model weights
- `train.py` – Main training script
- `requirements.txt` – Python dependencies

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Temesghen-Eyassu/AppliedDeep_Learning_for_ShipDetection.git
cd AppliedDeep_Learning_for_ShipDetection
git lfs pull   # ensure Git LFS is installed for large files

### 2. Install dependencies
```bash
pip install -r requirements.txt


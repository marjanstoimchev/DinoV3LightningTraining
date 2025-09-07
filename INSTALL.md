# Quick Installation Guide

## 🚀 One-Command Setup

```bash
# Clone, setup environment, and download weights
git clone <your-repo-url>
cd DinoV3Lightning_modified
./setup.sh
```

## 📋 Manual Setup

### 1. Environment
```bash
conda env create -f environment.yml
conda activate dinov3_lightning
```

### 2. Submodules
```bash
git submodule update --init --recursive
```

### 3. Weights (Optional)
```bash
mkdir -p dinov3_official_weights
wget -O dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    "https://dl.fbaipublicfiles.com/dinov3/dinov3_vits16_pretrain_lvd1689m.pth"
```

### 4. Test Installation
```bash
python -c "import torch, pytorch_lightning, omegaconf, datasets; print('✅ Installation successful')"
```

## ⚡ Quick Start

```bash
# Single GPU test
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output \
    --gpus 1 \
    --max-epochs 5 \
    --limit-train-batches 0.1

# Multi-GPU production
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_multigpu \
    --gpus 4 \
    --strategy ddp \
    --sampler-type distributed \
    --batch-size 128
```

## 🔧 Requirements

- **Python**: 3.11+
- **PyTorch**: 2.8.0+
- **CUDA**: 12.8+ (for GPU training)
- **Memory**: 16GB+ RAM, 8GB+ VRAM per GPU
- **Disk**: 10GB+ for dependencies and weights

## 📞 Need Help?

- Check [README.md](README.md) for detailed documentation
- See [docs/EXAMPLES.md](docs/EXAMPLES.md) for usage examples  
- Create an issue for problems
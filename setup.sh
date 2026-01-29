#!/bin/bash

# DINOv3 Lightning Setup Script
# This script sets up the complete environment for DINOv3 Lightning training

set -e  # Exit on any error

echo "🚀 Setting up DINOv3 Lightning Environment..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH"
    print_error "Please install Anaconda or Miniconda first"
    exit 1
fi

print_header "Step 1: Creating Conda Environment"

# Check if environment already exists
if conda env list | grep -q "dinov3_lightning"; then
    print_warning "Environment 'dinov3_lightning' already exists"
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing environment..."
        conda env remove -n dinov3_lightning -y
    else
        print_status "Using existing environment..."
        conda activate dinov3_lightning
        print_header "Step 2: Updating Dependencies"
        conda env update -f environment.yml
    fi
else
    print_status "Creating new conda environment from environment.yml..."
    conda env create -f environment.yml
fi

print_header "Step 3: Activating Environment"
print_status "Activating dinov3_lightning environment..."

# Source conda and activate environment
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate dinov3_lightning

print_header "Step 4: Initializing DINOv3 Submodule"
print_status "Initializing DINOv3 submodule..."
git submodule update --init --recursive

print_header "Step 5: Downloading DINOv3 Pretrained Weights"
print_status "Creating weights directory..."
mkdir -p dinov3_official_weights

# Download DINOv3 ViT-Small pretrained weights
WEIGHTS_FILE="dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
if [ ! -f "$WEIGHTS_FILE" ]; then
    print_status "Downloading DINOv3 ViT-Small pretrained weights..."
    wget -O "$WEIGHTS_FILE" "https://dl.fbaipublicfiles.com/dinov3/dinov3_vits16_pretrain_lvd1689m.pth" || {
        print_error "Failed to download pretrained weights"
        print_error "Please download manually from: https://dl.fbaipublicfiles.com/dinov3/"
    }
else
    print_status "Pretrained weights already exist: $WEIGHTS_FILE"
fi

print_header "Step 6: Verifying Installation"
print_status "Verifying Python packages..."

python -c "
import torch
import pytorch_lightning
import omegaconf
import datasets
import PIL
print('✅ Core packages verification successful')
print(f'PyTorch: {torch.__version__}')
print(f'PyTorch Lightning: {pytorch_lightning.__version__}')
print(f'OmegaConf: {omegaconf.__version__}')
print(f'Datasets: {datasets.__version__}')
print(f'Pillow: {PIL.__version__}')
" || {
    print_error "Package verification failed!"
    exit 1
}

print_header "Step 7: Creating Output Directories"
print_status "Creating necessary output directories..."
mkdir -p output_multi_gpu
mkdir -p lightning_output
mkdir -p logs

print_header "Step 8: Configuration Check"
print_status "Checking configuration file..."
if [ -f "configs/config_lightning_finetuning.yaml" ]; then
    print_status "✅ Configuration file found: configs/config_lightning_finetuning.yaml"
else
    print_error "❌ Configuration file not found!"
    exit 1
fi

print_header "Setup Complete! 🎉"
echo "=================================================="
print_status "Environment: dinov3_lightning"
print_status "Ready to train DINOv3 models!"
echo ""
print_status "Quick start commands:"
echo -e "${BLUE}# Activate environment${NC}"
echo "conda activate dinov3_lightning"
echo ""
echo -e "${BLUE}# Single GPU training${NC}"
echo "python src/training/train_dinov3_lightning.py \\"
echo "    --config-file configs/config_lightning_finetuning.yaml \\"
echo "    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \\"
echo "    --output-dir ./output \\"
echo "    --gpus 1 \\"
echo "    --sampler-type infinite"
echo ""
echo -e "${BLUE}# Multi-GPU training${NC}" 
echo "python src/training/train_dinov3_lightning.py \\"
echo "    --config-file configs/config_lightning_finetuning.yaml \\"
echo "    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \\"
echo "    --output-dir ./output_multigpu \\"
echo "    --gpus 4 \\"
echo "    --strategy ddp \\"
echo "    --sampler-type distributed \\"
echo "    --batch-size 128"
echo ""
print_status "For more information, see README.md"
echo "=================================================="



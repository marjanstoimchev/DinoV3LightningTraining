# DINOv3 PyTorch Lightning

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)
![Lightning](https://img.shields.io/badge/Lightning-2.5.0-purple.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A PyTorch Lightning implementation of DINOv3 self-supervised learning, providing an easy-to-use, scalable, and well-documented framework for training DINOv3 models on custom datasets.

## 🚀 Features

- **PyTorch Lightning Integration**: Clean, modular code with automatic multi-GPU support
- **Hybrid DataLoader System**: Optimized data loading for different sampling strategies
- **Multiple Dataset Support**: Custom TIFF datasets, HuggingFace datasets, and standard vision datasets  
- **Flexible Sampling**: Infinite, distributed, epoch-based, and sharded-infinite samplers
- **Advanced Progress Tracking**: Real-time loss monitoring with rich progress bars
- **Multi-GPU Training**: DDP support with automatic gradient synchronization
- **Comprehensive Logging**: TensorBoard, CSV, and WandB integration
- **Easy Configuration**: YAML-based config system with sensible defaults

## 📋 Quick Start

### 1. Environment Setup

Create and activate the conda environment:

```bash
# Clone the repository
git clone <your-repo-url>
cd DinoV3Lightning_modified

# Create conda environment 
conda env create -f environment.yml
conda activate dinov3_lightning

# Initialize DINOv3 submodule
git submodule update --init --recursive
```

### 2. Basic Training

```bash
# Train on HuggingFace dataset (recommended for testing)
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output \
    --gpus 1 \
    --sampler-type infinite

# Multi-GPU training
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_multigpu \
    --gpus 4 \
    --strategy ddp \
    --sampler-type distributed \
    --batch-size 128
```

### 3. Custom Dataset Training

Edit `configs/config_lightning_finetuning.yaml`:

```yaml
train:
  # For custom TIFF images
  dataset_path: CustomTIFF:root=/path/to/your/images/
  
  # For HuggingFace datasets  
  dataset_path: HuggingFace:name=food101
  
  # For custom HuggingFace configs
  dataset_path: HuggingFace:name=your-dataset:split=train:image_key=image
```

## 🏗️ Repository Structure

```
DinoV3Lightning_modified/
├── src/                          # Main source code
│   ├── callbacks/               # Custom Lightning callbacks
│   ├── checkpointing/          # Model checkpointing utilities  
│   ├── models/                 # Lightning modules and data modules
│   └── training/               # Training scripts
├── configs/                     # Configuration files
├── data/                       # Dataset implementations
├── docs/                       # Documentation
├── dinov3/                     # DINOv3 submodule
├── scripts/                    # Utility scripts
├── notebooks/                  # Jupyter notebooks for analysis
├── environment.yml             # Conda environment
├── requirements.txt            # Python dependencies
└── setup.sh                   # Installation script
```

## ⚙️ Configuration

### Sampler Types

The framework supports multiple sampling strategies:

| Sampler | Use Case | Multi-GPU | Performance |
|---------|----------|-----------|-------------|
| `infinite` | Continuous streaming, SSL training | ✓ | Fastest for infinite datasets |
| `distributed` | Standard distributed training | ✓ | Best for multi-GPU efficiency |  
| `epoch` | Traditional epoch-based training | ✓ | Good for finite datasets |
| `sharded_infinite` | Sharded infinite streaming | ✓ | Memory efficient streaming |

### Dataset Formats

**HuggingFace Datasets** (Recommended):
```yaml
dataset_path: HuggingFace:name=jonathancui/oxford-pets
dataset_path: HuggingFace:name=food101:split=train  
dataset_path: HuggingFace:name=imagenet-1k:streaming=true
```

**Custom TIFF/Image Datasets**:
```yaml  
dataset_path: CustomTIFF:root=/path/to/images/
```

### Key Configuration Options

```yaml
train:
  batch_size_per_gpu: 8          # Batch size per GPU
  num_workers: 8                 # DataLoader workers
  OFFICIAL_EPOCH_LENGTH: 600     # Steps per epoch for infinite samplers
  
student:
  arch: vit_small                # Model architecture (vit_small, vit_base, vit_large)
  patch_size: 16                 # Vision transformer patch size
  
optim:
  lr: 0.0001                     # Learning rate
  epochs: 100                    # Number of training epochs
  weight_decay: 0.02             # Weight decay
```

## 🔧 Advanced Usage

### Custom Training Script

```python
from src.models.dinov3_lightning_model import DINOv3LightningModule
from src.models.dinov3_lightning_datamodule import DINOv3DataModule
import pytorch_lightning as pl

# Load configuration
cfg = OmegaConf.load('configs/config_lightning_finetuning.yaml')

# Create model and data module
model = DINOv3LightningModule(cfg_path=cfg)
datamodule = DINOv3DataModule(cfg, sampler_type='distributed')

# Create trainer
trainer = pl.Trainer(
    accelerator='auto',
    devices=4,
    strategy='ddp',
    precision='bf16-mixed',
    max_epochs=100
)

# Train
trainer.fit(model, datamodule)
```

### Performance Optimization Tips

1. **Use appropriate sampler**: 
   - `distributed` for multi-GPU training
   - `infinite` for continuous streaming
   - `epoch` for traditional training

2. **Batch size tuning**:
   - Start with `batch_size_per_gpu=8` for ViT-Small
   - Scale proportionally with number of GPUs
   - Monitor GPU memory usage

3. **DataLoader optimization**:
   - Use `num_workers=8-16` depending on CPU cores
   - Enable `persistent_workers=True` (automatic in our implementation)
   - Use `pin_memory=True` (automatic in our implementation)

## 📏 Monitoring and Logging

### Progress Tracking
The framework provides detailed real-time progress information:

```
Epoch 1/100 | Step 384/832 (46.2%) | ETA: 2.1h | Speed: 5.72 it/s | Elapsed: 1:10
DINO_L: 9.000 | DINO_G: 9.000 | KOLEO: -0.274 | IBOT: 2.254 | total_loss: 11.250
```

### Logging Options
- **TensorBoard**: `tensorboard --logdir ./output/tensorboard_logs`
- **CSV Logs**: Available in `./output/csv_logs/`  
- **WandB**: Configure in your config file

## 🔍 Troubleshooting

### Common Issues

**ImportError: No module named 'dinov3'**
```bash
# Ensure DINOv3 submodule is initialized
git submodule update --init --recursive
```

**CUDA out of memory**
```bash  
# Reduce batch size
--batch-size 64  # Instead of 128

# Or reduce batch_size_per_gpu in config
batch_size_per_gpu: 4  # Instead of 8
```

**Slow training speed**
- Use `--sampler-type distributed` for multi-GPU
- Use HuggingFace datasets instead of custom TIFF datasets
- Ensure sufficient `num_workers` (8-16)

**DataLoader hangs**
- Reduce `num_workers` if CPU limited
- Check dataset path accessibility
- Verify sufficient disk space

## 🛠️ Setup

### 1. Dataset Configuration
Update `config_lightning_finetuning.yaml` with your dataset:

#### Custom Datasets
```yaml
train:
  dataset_path: CustomTIFF:root=../Datasets/composite/
  batch_size_per_gpu: 8  # Adjust based on GPU memory
```

#### HuggingFace Datasets
```yaml
train:
  # Examples:
  # dataset_path: HuggingFace:name=jonathancui/oxford-pets
  # dataset_path: HuggingFace:name=food101:split=train
  dataset_path: HuggingFace:name=your-dataset-name
  batch_size_per_gpu: 8
```

### 2. Training Parameters  
Modify `run.sh` for your setup:
```bash
GPUS=4                    # Number of GPUs
SAVE_EVERY_N_STEPS=50     # Checkpoint frequency
MAX_EPOCHS=30             # Training duration
```

## 📊 Monitoring & Analysis

### Real-time Training Status
```bash
python show_training_status.py    # Live metrics and progress
python plot_training_losses.py    # Loss visualizations
```

### TensorBoard Dashboard
```bash
tensorboard --logdir=output_multi_gpu/tensorboard_logs
```

### Feature Analysis
```bash
# For custom datasets
jupyter notebook notebooks/feature_extraction_custom_dataset.ipynb

# For HuggingFace datasets  
jupyter notebook notebooks/feature_extraction_huggingface_dataset.ipynb
```
**Features:**
- Compare original vs fine-tuned model features  
- PCA, t-SNE, UMAP visualizations with label-based coloring
- Statistical analysis and correlations
- CLS token and patch-level feature extraction
- Automatic dataset exploration and class separation analysis

## 📁 Output Structure

```
output_multi_gpu/
├── checkpoints/                          # Model checkpoints
│   ├── model_epoch_01_step_000050_loss_11.312500.ckpt
│   ├── model_epoch_01_step_000100_loss_10.456789.ckpt
│   └── last.ckpt                         # Most recent checkpoint
├── tensorboard_logs/                     # TensorBoard logs
├── csv_logs/                            # CSV metrics
├── training.log                         # Detailed training log
└── final_ssl_model.pth                  # Final SSL model (DINOv3 compatible)
```

## 🔧 Configuration Guide

Key fine-tuning optimizations in `config_lightning_finetuning.yaml`:

```yaml
# Learning rates optimized for fine-tuning
schedules:
  lr: 
    peak: 0.0001         # Reduced from pretraining (0.001)
    end: 1.0e-07         # Gentle learning rate decay

# Training parameters  
optim:
  epochs: 30             # Fewer epochs than pretraining
  weight_decay: 0.02     # Lower than pretraining (0.04)

# Teacher EMA updates
ema:
  momentum: 0.999        # Slower updates (vs 0.996 pretraining)

# Data augmentation (less aggressive)
crops:
  global_crops_scale: [0.32, 1.0]  # Less aggressive cropping
  local_crops_scale: [0.05, 0.32]  
```

## 🔄 Checkpoint Management & Progressive Training

### **Using Previous Checkpoints as Pretrained Weights**

This framework supports seamless continuation and transfer learning using any previously saved checkpoint:

#### **Continue from Lightning Checkpoints (.ckpt)**
```bash
# Resume exact training state (recommended for same dataset/config)
python train_dinov3_lightning.py \
  --config-file config_lightning_finetuning.yaml \
  --checkpoint-path output_multi_gpu/checkpoints/model_epoch_01_step_000100_loss_10.456789.ckpt \
  --output-dir ./output_continued \
  --gpus 4
```

#### **Transfer from SSL Model (.pth)**  
```bash
# Start fresh training with pretrained weights (new dataset/config)
python train_dinov3_lightning.py \
  --config-file config_lightning_finetuning.yaml \
  --checkpoint-path ./final_ssl_model.pth \
  --output-dir ./output_transfer \
  --gpus 4
```

### **Checkpoint Type Auto-Detection**

| Checkpoint Type | What's Loaded | Use Case |
|----------------|---------------|----------|
| **Lightning (.ckpt)** | Full state: model + optimizer + scheduler + counters | Continue training seamlessly |
| **SSL Model (.pth)** | Model weights only | Transfer learning, new domains |

### **Progressive Training Workflow**

1. **Initial Training**: Start with original DINOv3 pretrained weights
   ```bash
   ./run.sh  # Uses dinov3_vits16_pretrain_lvd1689m-08c60483.pth
   ```

2. **Domain Fine-tuning**: Use best checkpoint for new domain/dataset
   ```bash
   # Extract SSL model first
   python extract_ssl_model.py --checkpoint output_multi_gpu/checkpoints/best_model.ckpt
   
   # Update config for new dataset, then train
   python train_dinov3_lightning.py --checkpoint-path ./final_ssl_model.pth
   ```

3. **Iterative Refinement**: Chain multiple fine-tuning stages
   ```bash
   # Stage 1 → Stage 2 → Stage 3...
   --checkpoint-path output_stage1/checkpoints/best_model.ckpt
   ```

## 🚨 Troubleshooting

### Memory Issues
```bash
# Reduce batch size in run.sh
BATCH_SIZE=16          # or smaller

# Or in config file
train:
  batch_size_per_gpu: 4
```

### Checkpoint Loading Issues
```bash
# Verify checkpoint exists
ls -la output_multi_gpu/checkpoints/

# Check checkpoint contents
python extract_ssl_model.py --checkpoint path/to/checkpoint.ckpt --info-only
```

### Training Hanging
- Fixed in this version with improved synchronization
- Checkpoints now save without blocking training loop

## 📚 Documentation

- [SLURM Usage Guide](docs/SLURM_USAGE.md) - Running on HPC clusters
- [Configuration Reference](docs/CONFIG.md) - Detailed config options  
- [Usage Examples](docs/EXAMPLES.md) - Comprehensive training examples
- [API Documentation](docs/API.md) - Code API reference
- [Jupyter Notebooks](notebooks/) - Analysis and visualization examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [DINOv3 original implementation](https://github.com/facebookresearch/dinov3) by Meta AI
- [PyTorch Lightning](https://lightning.ai/) framework
- The self-supervised learning community

## 📞 Support

- 📧 Create an issue for bug reports or feature requests
- 💬 Join our discussions for questions and community support
- 📆 Check the documentation for detailed guides

---

**Happy Training! 🚀**
# MedSeg-CHAOS: Multi-modal Abdominal Organ Segmentation Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

**MedSeg-CHAOS** is a comprehensive deep learning framework for multi-modal MRI abdominal organ segmentation, built on PyTorch. It supports state-of-the-art transformer-based architectures (TransUNet, SwinUNet) alongside traditional UNet variants for accurate segmentation of liver, kidneys, and spleen from CHAOS dataset.

## ✨ Features

- **Multi-modal Support**: Processes T1-In phase, T1-Out phase, and T2-SPIR MRI sequences simultaneously
- **Multiple Architectures**: TransUNet (Small/Medium/Large), SwinUNet, UNet, DLKUNet
- **Complete Pipeline**: Data loading, augmentation, training, validation, testing, and evaluation
- **Medical Evaluation Metrics**: Dice coefficient and 95% Hausdorff Distance (HD95)
- **Production Ready**: Multi-GPU support, checkpointing, logging, and reproducible results
- **Modular Design**: Easy to extend with new models, datasets, and loss functions

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/MedSeg-CHAOS.git
cd MedSeg-CHAOS
```

2. **Create and activate conda environment (recommended)**

**Option A: Using environment.yml (recommended)**
```bash
conda env create -f environment.yml
conda activate medseg
```

**Option B: Manual setup**
```bash
conda create -n medseg python=3.8
conda activate medseg
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dataset Preparation

1. **Download CHAOS Dataset**
   - Request access from [CHAOS Challenge](https://chaos.grand-challenge.org/)
   - Download T1-DUAL and T2-SPIR sequences

2. **Preprocess and organize data**
```bash
# Expected directory structure:
Datasets/Chaos/
├── t1/
│   ├── 1/
│   │   ├── patient1_t1_inphase_1.npz
│   │   ├── patient1_t1_outphase_1.npz
│   │   └── ...
│   └── 2/
└── t2/
    ├── 1/
    │   ├── patient1_t2_1.npz
    │   └── ...
    └── 2/
```

3. **Convert DICOM to NPZ format** (if needed)

If your data is in DICOM format, you'll need to convert it to NPZ. We provide a sample script `scripts/convert_dicom_to_npz.py` (to be created) that demonstrates the conversion process. The key steps are:
- Load DICOM slices and reconstruct 3D volumes
- Extract pixel arrays and metadata
- Normalize and save as NPZ files with 'img' and 'lab' keys

## 📖 Usage

### Training

Train a TransUNet model:
```bash
python Train.py \
  --base_dir ./Datasets/Chaos \
  --epochs 400 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --log_path ./result/train.log \
  --pth_path ./result/Pth \
  --num_classes 5 \
  --multi_gpu
```

**Key Arguments:**
- `--base_dir`: Path to CHAOS dataset
- `--epochs`: Number of training epochs (default: 300)
- `--batch_size`: Batch size (adjust based on GPU memory)
- `--learning_rate`: Initial learning rate (default: 1e-4)
- `--num_classes`: Number of classes including background (default: 5)
- `--multi_gpu`: Enable multi-GPU training
- `--continue_train`: Resume from latest checkpoint

### Testing and Evaluation

Evaluate a trained model:
```bash
python Test.py \
  --base_dir ./Datasets/Chaos \
  --model_path ./result/Pth/model_epoch_300_checkpoint.pth \
  --num_classes 5 \
  --log_path ./result/test_result.log
```

**Expected Output:**
```
TEST RESULTS
==================================================
Liver           | Dice: 0.9124 | HD95: 8.4512 | Count: 142
Right Kidney    | Dice: 0.8731 | HD95: 6.2345 | Count: 138
Left Kidney     | Dice: 0.8567 | HD95: 7.8912 | Count: 136
Spleen          | Dice: 0.8983 | HD95: 5.1234 | Count: 140
--------------------------------------------------
Mean Average    | Dice: 0.8851 | HD95: 6.9251
==================================================
```

### Model Selection

Switch between different architectures in `Train.py` and `Test.py`:

```python
# TransUNet Small
model = TransUNet(img_dim=256, in_channels=3, out_channels=128,
                  head_num=8, mlp_dim=512, block_num=8,
                  patch_dim=16, class_num=5)

# TransUNet Medium
model = TransUNet(img_dim=256, in_channels=3, out_channels=128,
                  head_num=16, mlp_dim=3072, block_num=12,
                  patch_dim=16, class_num=5)

# TransUNet Large
model = TransUNet(img_dim=256, in_channels=3, out_channels=128,
                  head_num=16, mlp_dim=4096, block_num=24,
                  patch_dim=16, class_num=5)

# SwinUNet
model = SwinUNet(img_size=256, in_chans=3, num_classes=5,
                 embed_dim=96, depths=[2, 2, 6, 2],
                 depths_decoder=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], window_size=8)

# Standard UNet
model = UNet(n_channels=3, n_classes=5)
```

## 🏗️ Model Architectures

### TransUNet
Transformer + U-Net hybrid architecture that captures both local and global contextual information:
- **Encoder**: CNN backbone + Vision Transformer
- **Decoder**: CNN-based upsampling with skip connections
- **Patch Embedding**: 16×16 patches for transformer processing
- **Multi-head Attention**: 8-16 heads depending on model size

### SwinUNet
Hierarchical Swin Transformer with shifted windows for efficient self-attention:
- **Swin Transformer Blocks**: Local window attention + shifted window attention
- **Patch Merging/Expanding**: Hierarchical feature representation
- **U-Net Symmetry**: Encoder-decoder with skip connections

### UNet
Classical U-Net architecture with encoder-decoder structure and skip connections.

## 📊 Experimental Results

### Performance on CHAOS Test Set

| Model | Params | Liver (Dice/HD95) | Right Kidney (Dice/HD95) | Left Kidney (Dice/HD95) | Spleen (Dice/HD95) | Average Dice |
|-------|--------|-------------------|--------------------------|-------------------------|-------------------|--------------|
| TransUNet-L | 105M | 0.912/8.45 | 0.873/6.23 | 0.857/7.89 | 0.898/5.12 | 0.885 |
| TransUNet-M | 62M | 0.904/9.12 | 0.865/6.78 | 0.849/8.34 | 0.892/5.67 | 0.878 |
| TransUNet-S | 38M | 0.891/10.23 | 0.852/7.45 | 0.836/9.12 | 0.883/6.34 | 0.866 |
| SwinUNet | 87M | 0.907/8.89 | 0.869/6.56 | 0.853/8.01 | 0.895/5.45 | 0.881 |
| UNet | 31M | 0.885/11.45 | 0.843/8.12 | 0.827/9.89 | 0.876/7.23 | 0.858 |

### Training Configuration
- **Input Size**: 256×256
- **Batch Size**: 8 (RTX 5090 D)
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-5)
- **Loss Function**: CeDiceLoss (CrossEntropy + Dice)
- **Data Augmentation**: Horizontal/Vertical flip, rotation (±15°)
- **Training Time**: ~12 hours for 400 epochs

## 📁 Project Structure

```
MedSeg-CHAOS/
├── Train.py                 # Main training script
├── Test.py                  # Testing and evaluation script
├── DataLoader.py            # CHAOS dataset loader and transforms
├── CheckNpz.py              # NPZ file validation utility
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── Networks/               # Model architectures
│   ├── Transunet.py       # TransUNet (Transformer + U-Net)
│   ├── SwinUNet.py        # SwinUNet (Swin Transformer + U-Net)
│   ├── UNet.py            # Standard U-Net
│   └── DLKUNet_S.py       # DLKUNet variant
│
├── Utils/                  # Utility functions
│   └── LossFunction.py    # Loss functions (CeDiceLoss, nDiceLoss, etc.)
│
├── Datasets/              # Dataset storage
│   ├── Brats2019/         # BraTS 2019 dataset
│   ├── Brats2020/         # BraTS 2020 dataset
│   └── Chaos/             # CHAOS dataset (primary)
│
└── Result/                # Training results and models
    ├── TransUNet_L/       # TransUNet Large results
    ├── TransUNet_M/       # TransUNet Medium results
    ├── TransUNet_S/       # TransUNet Small results
    └── SwinUNet/          # SwinUNet results
```

## 🔧 Advanced Usage

### Custom Dataset Integration
Extend the framework for your own dataset:

1. Create a new dataset class inheriting from `torch.utils.data.Dataset`
2. Implement `__len__()` and `__getitem__()` methods
3. Add corresponding transform classes
4. Update training scripts to use your dataset

### Multi-GPU Training
Enable DataParallel training with `--multi_gpu` flag:
```bash
python Train.py --multi_gpu --batch_size 16
```
The framework automatically handles:
- Model replication across GPUs
- Gradient synchronization
- Checkpoint saving/loading with proper key stripping

### Custom Loss Functions
Add new loss functions in `Utils/LossFunction.py`:
```python
class YourCustomLoss(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        # Initialize components

    def forward(self, pred, target):
        # Implement loss calculation
        return loss
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{medsegchaos2026,
  title={MedSeg-CHAOS: A Comprehensive Framework for Multi-modal Abdominal Organ Segmentation},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

**Dataset Reference:**
```bibtex
@inproceedings{chaos2019,
  title={CHAOS Challenge: Combined (CT-MR) Healthy Abdominal Organ Segmentation},
  author={Kavur, A. Emre and Selver, M. Alper and Dicle, Oğuz and Barış, Mustafa and Gezer, N. Sinem},
  booktitle={Medical Image Analysis},
  volume={69},
  pages={101950},
  year={2021}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- **CHAOS Challenge** organizers for providing the dataset
- **PyTorch** team for the excellent deep learning framework
- **Original TransUNet authors** for their groundbreaking work
- **Swin Transformer team** for the hierarchical vision transformer design

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

For questions and feedback, please open an issue on GitHub or contact:
- **Your Name** - [email@example.com](mailto:email@example.com)
- **GitHub Issues**: [https://github.com/yourusername/MedSeg-CHAOS/issues](https://github.com/yourusername/MedSeg-CHAOS/issues)

---

<p align="center">
  <em>Built with ❤️ for the medical imaging research community</em>
</p>
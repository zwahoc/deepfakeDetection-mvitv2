# Deepfake Detection with MViTv2

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)


A deep learning project for detecting deepfake images using **Multiscale Vision Transformer v2 (MViTv2)** as the backbone model. This system classifies face images into **Real** and **Fake** categories with high accuracy.

## 📦 Dataset

This project uses the [Deepfake and Real Images dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) from Kaggle for training and validation.

## 🎯 Project Overview

Deepfakes pose serious risks in misinformation, fraud, privacy invasion, and trust in digital media. This project builds a prototype deepfake detection system that achieves **93.96% accuracy** on test data using a two-phase training approach:

- **Phase 1**: Head-only training (fine-tuning the classifier)
- **Phase 2**: Full model fine-tuning with unfrozen backbone layers

### 📊 Model Performance

| Metric | Real Images | Fake Images | Overall |
|--------|-------------|-------------|---------|
| Precision | 98.02% | 90.58% | 94.30% |
| Recall | 89.64% | 98.22% | 93.93% |
| F1-Score | 93.64% | 94.24% | 93.94% |
| **Accuracy** | - | - | **93.96%** |




## 🏗️ Project Structure

```
deepfakeDetection-mvitv2/
├── 📁 data/
│   ├── manifests/          # CSV files with image paths and labels
│   │   ├── train.csv       # Training dataset manifest (140k+ images)
│   │   ├── val.csv         # Validation dataset manifest (39k+ images)
│   │   └── test.csv        # Test dataset manifest (11k+ images)
│   └── raw/               # Raw image data
│       ├── train/         # Training images (fake/ and real/ subdirs)
│       ├── val/           # Validation images
│       └── test/          # Test images
├── 📁 notebooks/
│   └── deepfake.ipynb     # Main Jupyter notebook with complete pipeline
├── 📁 outputs/
│   ├── checkpoints/       # Trained model checkpoints
│   ├── figures/           # ROC curves, confusion matrices
│   ├── metrics/           # Training history and metrics
│   └── predictions/       # Model predictions and reports
├── .git/                 # Git version control
├── .gitignore            # Git ignore file
├── readme.md             # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (recommended for training)
- **Git** for cloning the repository

### Installation

#### For Windows Users

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/zwahoc/deepfakeDetection-mvitv2.git
   cd deepfakeDetection-mvitv2
   ```

2. **Create and activate a conda environment:**
   ```powershell
   conda create -n deepfake python=3.10 -y
   conda activate deepfake
   ```

3. **Install PyTorch with CUDA support:**
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install required packages:**
   ```powershell
   # Vision Transformer models (includes MViTv2)
   pip install timm
   
   # Image processing
   pip install opencv-python pillow albumentations
   
   # Utilities and ML libraries
   pip install matplotlib pandas scikit-learn tqdm
   
   # Jupyter Notebook support
   pip install notebook ipykernel ipywidgets
   
   # Optional: for explainability (Grad-CAM)
   pip install torchcam
   ```

5. **Set up Jupyter kernel:**
   ```powershell
   jupyter nbextension enable --py widgetsnbextension
   python -m ipykernel install --user --name=deepfake --display-name "Python (deepfake)"
   ```

#### For Mac Users

1. **Clone the repository:**
   ```bash
   git clone https://github.com/zwahoc/deepfakeDetection-mvitv2.git
   cd deepfakeDetection-mvitv2
   ```

2. **Create and activate a conda environment:**
   ```bash
   conda create -n deepfake python=3.10 -y
   conda activate deepfake
   ```

3. **Install PyTorch:**
   ```bash
   # For Mac with Apple Silicon (M1/M2)
   pip install torch torchvision torchaudio
   
   # For Intel Mac
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install required packages:**
   ```bash
   # Vision Transformer models (includes MViTv2)
   pip install timm
   
   # Image processing
   pip install opencv-python pillow albumentations
   
   # Utilities and ML libraries
   pip install matplotlib pandas scikit-learn tqdm
   
   # Jupyter Notebook support
   pip install notebook ipykernel ipywidgets
   
   # Optional: for explainability (Grad-CAM)
   pip install torchcam
   ```

5. **Set up Jupyter kernel:**
   ```bash
   jupyter nbextension enable --py widgetsnbextension
   python -m ipykernel install --user --name=deepfake --display-name "Python (deepfake)"
   ```

## 💾 Dataset Setup

1. **Prepare your dataset** with the following structure:
   ```
   data/raw/
   ├── train/
   │   ├── fake/    # Fake/deepfake images from kaggle's dataset in train folder
   │   └── real/    # Real/authentic images from kaggle's dataset in train folder
   ├── val/
   │   ├── fake/    # Fake/deepfake images from kaggle's dataset in validation folder
   │   └── real/    # Real/authentic images from kaggle's dataset in validation folder
   └── test/
       ├── fake/    # Fake/deepfake images from kaggle's dataset in test folder
       └── real/    # Real/authentic images from kaggle's dataset in test folder
   ```

2. **Generate manifest files** by running the notebook cells that create `train.csv`, `val.csv`, and `test.csv` in the `data/manifests/` directory.


## 📚 Usage

### Running the Complete Pipeline

All code for data loading, preprocessing, model training, evaluation, and visualization is contained in the Jupyter notebook:

1. **Start Jupyter Notebook:**
   ```bash
   # Windows
   jupyter notebook
   # Mac
   jupyter notebook
   ```

2. **Open `notebooks/deepfake.ipynb`** and select the "Python (deepfake)" kernel.

3. **Run the notebook cells sequentially** to:
   - Load and explore the dataset
   - Set up data augmentation and preprocessing
   - Create MViTv2 model with pretrained weights
   - Perform Phase 1 training (head-only)
   - Perform Phase 2 training (full fine-tuning)
   - Evaluate model performance
   - Generate visualizations and reports

### Training Configuration

The project uses a two-phase training approach, fully implemented in the notebook:

**Phase 1 (Head-only training):**
- Epochs: 8
- Batch size: 4
- Learning rate: 1e-3
- Early stopping patience: 3 epochs
- AMP (Automatic Mixed Precision): Enabled

**Phase 2 (Full fine-tuning):**
- Epochs: 10
- Batch size: 4
- Learning rates: 1e-5 (backbone), 1e-4 (head)
- Early stopping patience: 3 epochs
- AMP: Enabled

## 📈 Model Architecture

- **Backbone**: MViTv2-Tiny (pre-trained on ImageNet)
- **Input size**: 224×224 RGB images
- **Output**: 2-class classification (Real vs Fake)
- **Key features**:
  - Multiscale Vision Transformer architecture
  - Hierarchical feature representation
  - Efficient attention mechanisms
  - Dropout regularization (20%)

## 📊 Results and Outputs

The project generates comprehensive outputs:


### Checkpoints
- **Note:** Model checkpoint files are **not included** in this repository due to GitHub's file size limitations. You must train the model yourself to generate these files.
- Before training, **create the `checkpoints` directory** inside the `outputs` folder if it does not exist:
   ```bash
   mkdir -p outputs/checkpoints
   ```
- Example checkpoint files (created after training):
   - `mvitv2_phase1_best.pt` - Best Phase 1 model
   - `mvitv2_phase2_best.pt` - Best Phase 2 model (final)
   - Timestamped checkpoints for each epoch

### Visualizations
- ROC curves for validation and test sets
- Confusion matrices
- Training history plots

### Predictions
- CSV files with predictions and probabilities
- Classification reports with detailed metrics


## 🔧 Customization

### Model Configuration
Modify the model settings in the notebook:
```python
cfg["model_name"] = "mvitv2_tiny"  # or "mvitv2_small"
cfg["num_classes"] = 2
cfg["drop_rate"] = 0.2
```

### Training Parameters
Adjust training configurations:
```python
train_cfg = {
    "epochs": 8,
    "batch_size": 4,
    "lr_head": 1e-3,
    "patience": 3,
    # ... other parameters
}
```

## 🔍 Hardware Requirements


### Training Hardware (Current Specs)
- **LAPTOP**: Nitro 5 AN515-58
- **CPU**: Intel Core i5-12500H
- **RAM**: 32GB DDR4
- **GPU**: NVIDIA RTX3050 Laptop GPU (4GB VRAM)


### Minimum Requirements
- **RAM**: 8GB
- **Storage**: 10GB free space (for small datasets)
- **GPU**: NVIDIA GTX 1650 or equivalent (4GB VRAM)
Dedicated GPU strongly recommended for faster training. Training on CPU is possible but will be much slower; reduce batch size to fit into memory if using CPU or low VRAM GPU.


### Recommended Requirements
- **RAM**: 16GB+
- **Storage**: 50GB+ free space (for large datasets)
- **GPU**: NVIDIA RTX 3060 or better (6-12GB+ VRAM)

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size from 4 to 2 or 1
   - Enable gradient accumulation to simulate larger batches

2. **Slow Training on CPU**
   - Install CUDA-enabled PyTorch if you have an NVIDIA GPU
   - Consider using smaller model variants

3. **Import Errors**
   - Ensure all packages are installed in the correct conda environment
   - Verify the Jupyter kernel is using the right environment

### Performance Tips

- **Use SSD storage** for faster data loading
- **Set appropriate num_workers** in DataLoader (0 for Windows, 2-4 for Unix)
- **Enable AMP** for faster training with minimal accuracy loss
- **Use tensorboard** for monitoring training progress

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Contact

- **Author**: [zwahoc](https://github.com/zwahoc)
- **Project Link**: [https://github.com/zwahoc/deepfakeDetection-mvitv2](https://github.com/zwahoc/deepfakeDetection-mvitv2)

## 🙏 Acknowledgments

- [timm](https://github.com/rwightman/pytorch-image-models) for the MViTv2 implementation
- [PyTorch](https://pytorch.org/) for the deep learning framework
- The research community for advancing deepfake detection methods

---

**⚠️ Disclaimer**: This project is for educational and research purposes. Always consider ethical implications when working with deepfake detection technologies.
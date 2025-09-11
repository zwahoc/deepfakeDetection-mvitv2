# Multi-Model Deepfake Image Detection Platform

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive deep learning platform for detecting deepfake images featuring **multiple state-of-the-art models** including **MViTv2**, **EfficientNet-B3**, and **VGG-16+SRM**. This system includes both a complete training pipeline and an **interactive web application** for real-time deepfake detection with **up to 96.98% accuracy**.

## 📦 Dataset

This project uses the [Deepfake and Real Images dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) from Kaggle for training and validation.

## 🎯 Project Overview

Deepfakes pose serious risks in misinformation, fraud, privacy invasion, and trust in digital media. This project builds a complete **multi-model deepfake detection platform** that offers three state-of-the-art architectures with different strengths:

### 🧠 **Available Models**

| Model | Architecture | Accuracy | Best For |
|-------|-------------|----------|----------|
| **VGG-16 + SRM** | CNN + Spatial Rich Model | **96.98%** | High accuracy, forensic analysis |
| **MViTv2-Tiny** | Vision Transformer | **93.96%** | Balanced performance, modern architecture |
| **EfficientNet-B3** | Efficient CNN | **91.30%** | Fast inference, resource efficiency |

### 🚀 **Enhanced Web Application**

The platform features a **sophisticated Streamlit web application** (`app.py`) that provides:

- **🔄 Multi-model selection**: Choose between VGG-16+SRM, MViTv2, or EfficientNet-B3
- **📤 Multi-image upload**: Upload and analyze multiple images simultaneously
- **🎛️ Advanced controls**: Configurable decision threshold and uncertainty detection
- **🌡️ Temperature calibration**: Advanced probability calibration for MViTv2 (better confidence estimates)
- **📱 Modern carousel UI**: Navigate through multiple images with smooth transitions
- **🔧 Cross-platform**: Supports CUDA, MPS (Apple Silicon), and CPU devices
- **⚖️ Uncertainty quantification**: Flags predictions when models are uncertain
- **📊 Real-time metrics**: Live performance statistics for each model

### 🔗 **Quick Demo**

```bash
# Launch the web application
streamlit run app.py
```

Visit `http://localhost:8501` to access the interactive interface!

### 📊 Model Performance

#### **VGG-16 + SRM (Best Overall)**
| Metric | Real Images | Fake Images | Overall |
|--------|-------------|-------------|---------|
| Precision | 96.44% | 96.44% | **96.44%** |
| Recall | 97.54% | 97.54% | **97.54%** |
| F1-Score | 96.99% | 96.99% | **96.99%** |
| **Accuracy** | - | - | **96.98%** |

#### **MViTv2-Tiny (Balanced)**
| Metric | Real Images | Fake Images | Overall |
|--------|-------------|-------------|---------|
| Precision | 98.02% | 90.58% | 94.30% |
| Recall | 89.64% | 98.22% | 93.93% |
| F1-Score | 93.64% | 94.24% | 93.94% |
| **Accuracy** | - | - | **93.96%** |

#### **EfficientNet-B3 (Efficient)**
| Metric | Real Images | Fake Images | Overall |
|--------|-------------|-------------|---------|
| Precision | 92.00% | 92.00% | **92.00%** |
| Recall | 91.00% | 91.00% | **91.00%** |
| F1-Score | 91.00% | 91.00% | **91.00%** |
| **Accuracy** | - | - | **91.30%** |




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
│   └── deepfake.ipynb     # Complete training pipeline notebook
├── 📁 outputs/
│   ├── checkpoints/       # Trained model checkpoints
│   │   ├── mvitv2_phase2_best.pt    # MViTv2 model (default)
│   │   ├── best_efficientnet_b3.pth # EfficientNet-B3 model
│   │   └── deepfake_cnn.pt          # VGG-16+SRM model
│   ├── calibration/       # Temperature calibration files
│   ├── figures/           # ROC curves, confusion matrices
│   ├── metrics/           # Training history and metrics
│   └── predictions/       # Model predictions and reports
├── 📁 B3-checkpoint/      # EfficientNet-B3 model checkpoints
├── 📁 VGG16-checkpoint/   # VGG-16+SRM model checkpoints
├── 📱 app.py              # Advanced multi-model Streamlit application
├── 📋 requirements.txt    # Python dependencies
├── 🐍 environment.yml     # Conda environment configuration
├── 📄 readme.md           # Project documentation
└── 🔧 .gitignore          # Git ignore patterns
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **CUDA/MPS/CPU**: Supports NVIDIA GPUs, Apple Silicon (M1/M2), and CPU fallback
- **Git** for cloning the repository

### Easy Installation (Recommended)

#### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/zwahoc/deepfakeDetection-mvitv2.git
cd deepfakeDetection-mvitv2

# Create environment from file
conda env create -f environment.yml
conda activate deepfake

# Launch the web app
streamlit run app.py
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/zwahoc/deepfakeDetection-mvitv2.git
cd deepfakeDetection-mvitv2

# Create virtual environment
python -m venv deepfake-env
source deepfake-env/bin/activate  # On Windows: deepfake-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the web app
streamlit run app.py
```

### 🌐 Web Application Usage

1. **Launch the app**: `streamlit run app.py`
2. **Open browser**: Navigate to `http://localhost:8501`
3. **Select model**: Choose from VGG-16+SRM, MViTv2, or EfficientNet-B3 in the sidebar
4. **Upload images**: Use the file uploader to select face images (supports JPG, PNG, JPEG, WebP, BMP)
5. **Adjust settings**: Configure threshold and calibration options
6. **View results**: See predictions, confidence scores, and technical details
7. **Navigate**: Use carousel controls to browse through multiple uploaded images
8. **Analyze performance**: View live model performance metrics

### 🎛️ **Application Features Breakdown**

#### **Model Selection**
- **VGG-16 + SRM**: Best for forensic-level analysis with 96.98% accuracy
- **MViTv2-Tiny**: Modern transformer with temperature calibration support
- **EfficientNet-B3**: Fast and efficient for real-time applications

#### **Advanced Controls**
- **Decision Threshold**: Adjust sensitivity (0.30-0.90, default: 0.50)
- **Temperature Calibration**: Available for MViTv2 to improve confidence estimates
- **Uncertainty Zone**: Flags predictions with low confidence (±5% around 50% by default)

#### **User Interface**
- **Carousel View**: Navigate through multiple images with prev/next controls
- **Progress Bars**: Visual representation of real/fake probabilities
- **Auto-scrolling**: Smooth navigation to results after upload
- **Responsive Design**: Works on desktop and mobile browsers

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

### 🌐 Web Application (Recommended)

The easiest way to use the multi-model deepfake detection platform:

```bash
streamlit run app.py
```

**Key Features:**
- **Multi-model support**: Switch between VGG-16+SRM, MViTv2, and EfficientNet-B3
- **Interactive interface**: Upload and analyze images through a web browser
- **Batch processing**: Handle multiple images simultaneously with carousel navigation
- **Real-time results**: See predictions and confidence scores instantly
- **Advanced settings**: Configure decision threshold, temperature calibration, and uncertainty detection
- **Cross-platform support**: Works on Windows, macOS, and Linux
- **Multi-device support**: Automatically detects and uses CUDA, MPS, or CPU
- **Performance metrics**: Live accuracy, precision, recall, and F1-score displays

### 📓 Training Pipeline (Advanced Users)

For training your own models or exploring the complete pipeline:

1. **Start Jupyter Notebook:**
   ```bash
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

### 🎛️ Application Configuration

The web application supports extensive configuration options:

#### **Model Selection**
- **VGG-16 + SRM**: Custom forensic architecture with Spatial Rich Model preprocessing
- **MViTv2-Tiny**: Vision Transformer with hierarchical attention mechanisms  
- **EfficientNet-B3**: Compound scaling CNN optimized for efficiency

#### **Inference Settings**
- **Decision Threshold**: Adjust the threshold for fake/real classification (0.30-0.90, default: 0.5)
- **Temperature Calibration**: Enable/disable temperature scaling for MViTv2 (improves probability estimates)
- **Uncertainty Zone**: Configure the range around 50% to flag uncertain predictions (0.01-0.20, default: 0.05)
- **Batch Processing**: Automatic chunking for large image batches (max 64 images per batch)

#### **Input Processing**
- **Supported Formats**: PNG, JPG, JPEG, WebP, BMP
- **Automatic Preprocessing**: Image resizing, normalization, and tensor conversion
- **Multi-image Upload**: Simultaneous processing of multiple images

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

## 📈 Model Architectures

### **1. VGG-16 + SRM (Forensic Model)**
- **Backbone**: VGG-16 feature extractor + custom head
- **Special Feature**: Spatial Rich Model (SRM) preprocessing for forensic analysis
- **Input size**: 244×244 RGB images
- **Output**: Single logit (sigmoid activation)
- **Architecture**: VGG features → Global Average Pooling → BatchNorm → Dropout(0.4) → Linear(512→1)

### **2. MViTv2-Tiny (Vision Transformer)**
- **Backbone**: Multiscale Vision Transformer v2 (pre-trained on ImageNet)
- **Input size**: 224×224 RGB images  
- **Output**: 2-class classification (Real vs Fake)
- **Key features**:
  - Hierarchical multiscale attention mechanisms
  - Efficient transformer architecture
  - Temperature calibration support
  - Dropout regularization (20%)

### **3. EfficientNet-B3 (Efficient CNN)**
- **Backbone**: EfficientNet-B3 (compound scaling architecture)
- **Input size**: 300×300 RGB images
- **Output**: 2-class classification
- **Architecture**: EfficientNet backbone → Dropout(0.3) → Linear(1536→512) → ReLU → BatchNorm → Dropout(0.2) → Linear(512→2)
- **Key features**:
  - Compound scaling (depth, width, resolution)
  - Mobile-optimized design
  - Fast inference speed

## 📊 Results and Outputs

The project generates comprehensive outputs and provides multiple ways to access results:

### 🌐 **Web Application Results**
- **Real-time predictions**: Instant classification with confidence scores
- **Visual feedback**: Color-coded results (green for real, red for fake)
- **Progress bars**: Visual representation of probability distributions
- **Technical details**: Device info, model parameters, and processing statistics
- **Uncertainty detection**: Flags when model confidence is low
- **Batch results**: Gallery view for multiple image analysis

### 💾 **Training Outputs**

#### Model Checkpoints
> **Note:** Model checkpoints are **not included** in the repository to keep it lightweight and within GitHub limits.

- 📥 **Download all pretrained model checkpoints here:**  
  **OneDrive:** https://1drv.ms/f/c/ce853d1a7c00dfcc/EjuEaS0VDypMgBsEwnm89UYByVgU3AN5xpZ5w_26vW2-zw?e=gQ3Amm

- **Required checkpoint structure:**
  ```
  outputs/checkpoints/
  ├── mvitv2_phase2_best.pt      # MViTv2 model (93.96% accuracy)
  
  B3-checkpoint/
  ├── best_efficientnet_b3.pth   # EfficientNet-B3 model (91.30% accuracy)
  
  VGG16-checkpoint/
  ├── deepfake_cnn.pt            # VGG-16+SRM model (96.98% accuracy)
  ```

- **App behavior**: The application will automatically detect available models and show them in the model selector. If a checkpoint is missing, that model will be unavailable until the checkpoint is added.

- **Compatibility**: Each model has its own checkpoint format and loading mechanism built into the app.

#### Calibration
- **Temperature scaling**: Calibrated temperature values for improved probability estimates
- **Calibration file**: `outputs/calibration/temperature.json`

#### Visualizations
- ROC curves for validation and test sets
- Confusion matrices with detailed breakdowns
- Training history plots and metrics

#### Predictions
- CSV files with predictions and probabilities
- Classification reports with detailed metrics
- Per-image analysis results


## 🔧 Advanced Configuration

### Web Application Advanced Settings

The Streamlit app provides extensive configuration through the sidebar and `MODEL_REGISTRY`:

```python
# Model registry configuration (app.py)
MODEL_REGISTRY = {
    "mvitv2_tiny": {
        "display_name": "MViTv2-Tiny",
        "backend": "timm",
        "input_size": 224,
        "calibration_supported": True,  # Temperature scaling available
    },
    "efficientnet_b3": {
        "display_name": "EfficientNet-B3", 
        "backend": "torchvision",
        "input_size": 300,
        "calibration_supported": False,  # No temperature scaling
    },
    "vgg16_srm": {
        "display_name": "VGG-16 + SRM",
        "backend": "custom_vgg16_srm",
        "input_size": 244,
        "calibration_supported": False,  # No temperature scaling
    }
}

# Runtime configuration
threshold = 0.5         # Decision threshold (0.30-0.90)
calib_on = True         # Enable temperature calibration (MViTv2 only)
uncert_band = 0.05      # Uncertainty zone (±5% around 50%)
```

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

### Device Selection
The application automatically selects the best available device:
```python
device = (
    torch.device("mps") if torch.backends.mps.is_available()      # Apple Silicon
    else torch.device("cuda") if torch.cuda.is_available()        # NVIDIA GPU
    else torch.device("cpu")                                      # CPU fallback
)
```

## 🔍 Hardware Requirements

### 🌐 **Web Application (Inference)**
- **Minimum**: 4GB RAM, any modern CPU
- **Recommended**: 8GB+ RAM, dedicated GPU or Apple Silicon
- **Storage**: 2GB for model and dependencies
- **Supported devices**: 
  - ✅ NVIDIA GPUs (CUDA)
  - ✅ Apple Silicon (MPS) - M1/M2/M3
  - ✅ CPU (Intel/AMD/ARM)

### 🏋️ **Training Pipeline**
#### Current Development Hardware
- **LAPTOP**: Nitro 5 AN515-58
- **CPU**: Intel Core i5-12500H
- **RAM**: 32GB DDR4
- **GPU**: NVIDIA RTX3050 Laptop GPU (4GB VRAM)

#### Minimum Training Requirements
- **RAM**: 8GB
- **Storage**: 10GB free space (for small datasets)
- **GPU**: NVIDIA GTX 1650 or equivalent (4GB VRAM)
- **Note**: Training on CPU is possible but much slower; reduce batch size for low-memory systems

#### Recommended Training Requirements
- **RAM**: 16GB+
- **Storage**: 50GB+ free space (for large datasets)
- **GPU**: NVIDIA RTX 3060 or better (6-12GB+ VRAM)
- **Alternative**: Apple Silicon M1 Pro/Max/Ultra for Mac users

## 🐛 Troubleshooting

### 🌐 Web Application Issues

1. **App won't start**
   ```bash
   # Check if Streamlit is installed
   streamlit --version
   
   # If missing, install dependencies
   pip install -r requirements.txt
   ```

2. **Model loading errors**
   - Ensure the required checkpoint files exist:
     - `outputs/checkpoints/mvitv2_phase2_best.pt` (MViTv2)
     - `B3-checkpoint/best_efficientnet_b3.pth` (EfficientNet-B3)
     - `VGG16-checkpoint/deepfake_cnn.pt` (VGG-16+SRM)
   - Download missing checkpoints from OneDrive link above
   - Check if the model architecture matches the checkpoint
   - Verify sufficient disk space and memory

3. **Model not appearing in selector**
   - Check if the corresponding checkpoint file exists
   - Verify the checkpoint path in `MODEL_REGISTRY` (app.py)
   - Restart the application after adding new checkpoints

4. **Slow inference**
   - Check which device is being used (displayed in app sidebar)
   - For NVIDIA GPUs: ensure CUDA is properly installed
   - For Apple Silicon: verify MPS is available
   - Consider using a model with smaller input size (VGG-16: 244px vs EfficientNet-B3: 300px)

4. **Memory errors**
   - Reduce the number of images uploaded simultaneously
   - Restart the application to clear memory
   - Close other resource-intensive applications

### 🏋️ Training Pipeline Issues

1. **CUDA Out of Memory**
   - Reduce batch size from 4 to 2 or 1
   - Enable gradient accumulation to simulate larger batches
   - Use smaller model variants (mvitv2_tiny instead of mvitv2_small)

2. **Slow Training on CPU**
   - Install CUDA-enabled PyTorch if you have an NVIDIA GPU
   - Consider using Google Colab or cloud platforms for GPU access
   - Use data loading optimizations (num_workers=0 for debugging)

3. **Import Errors**
   - Ensure all packages are installed: `pip install -r requirements.txt`
   - Verify the correct conda environment is activated
   - Check Python version compatibility (3.10+ required)

4. **Dataset Loading Issues**
   - Verify the data structure matches the expected format
   - Check file permissions and paths
   - Ensure sufficient disk space for dataset extraction

### 💡 Performance Tips

#### For Web Application:
- **Browser**: Use Chrome or Firefox for best compatibility
- **Upload**: Resize large images before upload for faster processing
- **Model selection**: Use VGG-16+SRM for highest accuracy, EfficientNet-B3 for speed
- **Batch size**: Upload 5-10 images at a time for optimal performance
- **Memory**: Restart app periodically for long sessions to clear memory

#### For Training:
- **Storage**: Use SSD storage for faster data loading
- **Memory**: Set appropriate `num_workers` in DataLoader (0 for Windows, 2-4 for Unix)
- **Precision**: Enable AMP for faster training with minimal accuracy loss
- **Monitoring**: Use the notebook's built-in progress bars and metrics

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
- [Streamlit](https://streamlit.io/) for the web application framework
- [Albumentations](https://albumentations.ai/) for data augmentation
- The research community for advancing deepfake detection methods
- [Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) for providing the dataset

## 🌟 Features Summary

### ✅ **What's Included**
- � **Three state-of-the-art models** with up to 96.98% accuracy
- 🧠 **Multi-model platform** supporting VGG-16+SRM, MViTv2, and EfficientNet-B3
- 🌐 **Advanced web application** with model selection and carousel UI
- 📚 **Complete training pipeline** in Jupyter notebook  
- 🔧 **Cross-platform support** (Windows, macOS, Linux)
- 🚀 **Multi-device acceleration** (CUDA, MPS, CPU)
- 📊 **Comprehensive evaluation** metrics and live performance displays
- 🌡️ **Temperature calibration** for MViTv2 (better confidence estimates)
- ⚖️ **Uncertainty quantification** for reliable predictions
- 📱 **Batch processing** with smooth carousel navigation
- 🎨 **Modern responsive UI** with auto-scrolling and progress bars
- 🔄 **Real-time model switching** without app restart

### 🔮 **Potential Enhancements**
- **Multi-model ensemble**: Combine predictions from all three models for improved accuracy
- **Real-time video processing**: Extend to video deepfake detection
- **REST API endpoint**: Programmatic access for integration
- **Additional architectures**: Add ResNet, DenseNet, or custom forensic models
- **Mobile app development**: Native iOS/Android applications
- **Cloud deployment**: AWS/Azure hosting with auto-scaling
- **Advanced preprocessing**: Face detection and alignment pipeline

---

**⚠️ Disclaimer**: This tool is for educational and research purposes only. 
Deepfake detection is an evolving field, and no model is 100% accurate. 
Always verify important content through multiple sources and consider the ethical implications of deepfake technology.
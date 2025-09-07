# Deepfake Image Detection with MViTv2

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive deep learning project for detecting deepfake images using **Multiscale Vision Transformer v2 (MViTv2)** as the backbone model. This system includes both a complete training pipeline and a **interactive web application** for real-time deepfake detection with **93.96% accuracy**.

## 📦 Dataset

This project uses the [Deepfake and Real Images dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) from Kaggle for training and validation.

## 🎯 Project Overview

Deepfakes pose serious risks in misinformation, fraud, privacy invasion, and trust in digital media. This project builds a complete deepfake detection system that achieves **93.96% accuracy** on test data using a two-phase training approach:

- **Phase 1**: Head-only training (fine-tuning the classifier)
- **Phase 2**: Full model fine-tuning with unfrozen backbone layers

### 🚀 **NEW: Interactive Web Application**

This project now includes a **Streamlit web application** (`app.py`) that provides:

- **📤 Multi-image upload**: Upload and analyze multiple images simultaneously
- **🎛️ Adjustable controls**: Configurable decision threshold and uncertainty detection
- **🌡️ Temperature calibration**: Advanced probability calibration for better confidence estimates
- **📱 Modern UI**: Carousel view, progress bars, and responsive design
- **🔧 Cross-platform**: Supports CUDA, MPS (Apple Silicon), and CPU devices
- **⚖️ Uncertainty quantification**: Flags predictions when the model is uncertain

### 🔗 **Quick Demo**

```bash
# Launch the web application
streamlit run app.py
```

Visit `http://localhost:8501` to access the interactive interface!

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
│   └── deepfake.ipynb     # Complete training pipeline notebook
├── 📁 outputs/
│   ├── checkpoints/       # Trained model checkpoints
│   ├── calibration/       # Temperature calibration files
│   ├── figures/           # ROC curves, confusion matrices
│   ├── metrics/           # Training history and metrics
│   └── predictions/       # Model predictions and reports
├── 📱 app.py              # Streamlit web application
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
3. **Upload images**: Use the file uploader to select face images
4. **Adjust settings**: Configure threshold and calibration in the sidebar
5. **View results**: See predictions, confidence scores, and technical details
6. **Batch processing**: Upload multiple images for batch analysis

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

The easiest way to use the deepfake detection system:

```bash
streamlit run app.py
```

**Features:**
- **Interactive interface**: Upload and analyze images through a web browser
- **Batch processing**: Handle multiple images simultaneously
- **Real-time results**: See predictions and confidence scores instantly
- **Adjustable settings**: Configure decision threshold and uncertainty detection
- **Temperature calibration**: Enable advanced probability calibration
- **Cross-platform support**: Works on Windows, macOS, and Linux
- **Multi-device support**: Automatically detects and uses CUDA, MPS, or CPU

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

The web application supports several configuration options:

- **Decision Threshold**: Adjust the threshold for fake/real classification (default: 0.5)
- **Temperature Calibration**: Enable/disable temperature scaling for better probability estimates
- **Uncertainty Zone**: Configure the range around 50% to flag uncertain predictions
- **Batch Size**: Automatically handles batch processing for multiple images

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

The project generates comprehensive outputs and provides multiple ways to access results:

### 🌐 **Web Application Results**
- **Real-time predictions**: Instant classification with confidence scores
- **Visual feedback**: Color-coded results (green for real, red for fake)
- **Progress bars**: Visual representation of probability distributions
- **Technical details**: Device info, model parameters, and processing statistics
- **Uncertainty detection**: Flags when model confidence is low
- **Batch results**: Gallery view for multiple image analysis

### 💾 **Training Outputs**

#### Checkpoints
- **Available models**: Pre-trained checkpoints included for immediate use
- **Model files**:
   - `mvitv2_phase1_best.pt` - Best Phase 1 model
   - `mvitv2_phase2_best.pt` - Best Phase 2 model (recommended)
   - Timestamped checkpoints for each epoch

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

### Web Application Settings

The Streamlit app provides several configuration options accessible through the sidebar:

```python
# Example configuration in app.py
threshold = 0.5  # Decision threshold for classification
calib_on = True  # Enable temperature calibration
uncert_band = 0.05  # Uncertainty zone around 50%
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
   - Ensure the checkpoint file exists: `outputs/checkpoints/mvitv2_phase2_best.pt`
   - Check if the model architecture matches the checkpoint
   - Verify sufficient disk space and memory

3. **Slow inference**
   - Check which device is being used (displayed in the app)
   - For NVIDIA GPUs: ensure CUDA is properly installed
   - For Apple Silicon: verify MPS is available
   - Consider reducing image resolution or batch size

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
- **Batch size**: Upload 5-10 images at a time for optimal performance

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
- 🎯 **93.96% accuracy** deepfake detection model
- 🌐 **Interactive web application** with modern UI
- 📚 **Complete training pipeline** in Jupyter notebook
- 🔧 **Cross-platform support** (Windows, macOS, Linux)
- 🚀 **Multi-device acceleration** (CUDA, MPS, CPU)
- 📊 **Comprehensive evaluation** metrics and visualizations
- 🌡️ **Temperature calibration** for better confidence estimates
- ⚖️ **Uncertainty quantification** for reliable predictions
- 📱 **Batch processing** for multiple images
- 🎨 **Modern UI** with carousel view and responsive design

### 🔮 **Potential Enhancements**
- Real-time video processing
- API endpoint for integration
- Model ensemble for improved accuracy
- Additional deepfake detection techniques
- Mobile app development
- Cloud deployment options

---

**⚠️ Disclaimer**: This tool is for educational and research purposes only. 
Deepfake detection is an evolving field, and no model is 100% accurate. 
Always verify important content through multiple sources and consider the ethical implications of deepfake technology.
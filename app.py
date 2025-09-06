import streamlit as st
import torch
import timm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import PIL.Image as Image

# Configure page
st.set_page_config(
    page_title="Deepfake Detection with MViTv2",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
PROJECT_ROOT = Path(__file__).parent
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "mvitv2_phase2_best.pt"

@st.cache_resource
def load_model():
    """Load the trained MViTv2 model."""
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    
    # Model configuration (same as training)
    model_config = {
        "model_name": "mvitv2_tiny",
        "num_classes": 2,
        "drop_rate": 0.2
    }
    
    # Create model WITHOUT downloading ImageNet weights (we'll load our checkpoint)
    model = timm.create_model(
        model_config["model_name"],
        pretrained=False,
        num_classes=model_config["num_classes"],
        drop_rate=model_config["drop_rate"],
    )
    
    # Load trained weights (handle both dict formats)
    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state, strict=False)
        st.success(f"✅ Loaded trained model from {CHECKPOINT_PATH.name}")
    else:
        st.error(f"❌ Model checkpoint not found at {CHECKPOINT_PATH}")
        st.stop()
    
    model.eval()
    model.to(device)
    return model, device

@st.cache_data
def get_transforms():
    """Get image preprocessing transforms (same as training)."""
    # ImageNet normalization (same as training)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_size = 224
    
    transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    
    return transforms

def preprocess_image(image, transforms):
    """Preprocess uploaded image for model inference."""
    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure RGB format
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Already RGB
        pass
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # RGBA to RGB
        image = image[:, :, :3]
    else:
        st.error("Unsupported image format. Please upload RGB or RGBA images.")
        return None
    
    # Apply transforms
    transformed = transforms(image=image)
    tensor_image = transformed["image"]
    
    # Add batch dimension
    tensor_image = tensor_image.unsqueeze(0)
    
    return tensor_image

def predict_deepfake(model, device, image_tensor, threshold: float = 0.5):
    """Make prediction on preprocessed image with adjustable threshold."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Get model output (logits)
        logits = model(image_tensor)
        
        # Convert to probabilities
        if logits.size(1) == 2:  # 2-class output
            probs = torch.softmax(logits, dim=1)
            fake_prob = probs[0, 1].cpu().item()  # Probability of being fake
            real_prob = probs[0, 0].cpu().item()  # Probability of being real
        else:  # Single output (binary classification)
            prob = torch.sigmoid(logits).squeeze().cpu().item()
            fake_prob = float(prob)
            real_prob = float(1.0 - prob)
        
        # Determine prediction using provided threshold
        prediction = "FAKE" if fake_prob >= threshold else "REAL"
        confidence = fake_prob if prediction == "FAKE" else real_prob
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "fake_probability": float(fake_prob),
            "real_probability": float(real_prob)
        }

def main():
    # Title and description
    st.title("🔍 Deepfake Detection with MViTv2")
    st.markdown("""
    This application uses a **Multiscale Vision Transformer v2 (MViTv2)** model to detect deepfake images.
    The model has been trained to classify face images as **Real** or **Fake** with **93.96% accuracy**.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.markdown("""
        **Model**: MViTv2-Tiny  
        **Training**: Two-phase approach  
        **Accuracy**: 93.96%  
        **Input Size**: 224×224 RGB  
        **Classes**: Real vs Fake  
        """)
        
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload a face image (JPG, PNG, JPEG)
        2. The model will analyze the image
        3. Results show prediction and confidence
        4. Green = Real, Red = Fake
        """)
        
        st.header("⚠️ Notes")
        st.markdown("""
        - Works best with clear face images
        - Model trained on specific dataset
        - For research/educational purposes
        - Not 100% accurate - use with caution
        """)
        
        threshold = st.slider("Decision threshold (fake if ≥ threshold)", min_value=0.30, max_value=0.90, value=0.50, step=0.01)
    
    # Load model
    model, device = load_model()
    transforms = get_transforms()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a face image to analyze"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Show image details
            st.caption(f"📊 Image size: {image.size[0]}×{image.size[1]} pixels")
    
    with col2:
        st.header("🎯 Detection Results")
        
        if uploaded_file is not None:
            with st.spinner("🔍 Analyzing image..."):
                # Preprocess image
                image_tensor = preprocess_image(image, transforms)
                
                if image_tensor is not None:
                    # Make prediction
                    result = predict_deepfake(model, device, image_tensor, threshold=threshold)
                    
                    # Display results
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    fake_prob = result["fake_probability"]
                    real_prob = result["real_probability"]
                    
                    # Main prediction with color coding
                    if prediction == "REAL":
                        st.success(f"✅ **{prediction}**")
                    else:
                        st.error(f"❌ **{prediction}**")
                    
                    # Confidence score
                    st.metric(
                        label="Confidence",
                        value=f"{confidence:.1%}",
                        help="How confident the model is in its prediction"
                    )
                    
                    # Probability breakdown
                    st.subheader("📊 Probability Breakdown")
                    
                    # Real probability bar
                    st.markdown("**Real Image Probability:**")
                    st.progress(int(real_prob * 100))
                    st.caption(f"{real_prob:.1%}")
                    
                    # Fake probability bar
                    st.markdown("**Fake Image Probability:**")
                    st.progress(int(fake_prob * 100))
                    st.caption(f"{fake_prob:.1%}")
                    
                    # Additional info
                    st.subheader("🔬 Technical Details")
                    st.markdown(f"""
                    - **Model**: MViTv2-Tiny
                    - **Device**: {device.type.upper()}
                    - **Input Shape**: {list(image_tensor.shape)}
                    - **Preprocessing**: Resize to 224×224, ImageNet normalization
                    - **Decision Threshold**: {threshold:.2f}
                    """)
                    
                else:
                    st.error("Failed to preprocess image")
        else:
            st.info("👆 Upload an image to see detection results")
            
            # Show example
            st.subheader("💡 Example")
            st.markdown("""
            The model analyzes facial features and artifacts to determine if an image is:
            - **Real**: Authentic photograph of a person
            - **Fake**: AI-generated or manipulated image (deepfake)
            """)
    
    # Additional information
    st.markdown("---")
    st.subheader("📈 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Accuracy", "93.96%")
    with col2:
        st.metric("Precision (Real)", "98.02%")
    with col3:
        st.metric("Recall (Real)", "89.64%")
    with col4:
        st.metric("F1-Score", "93.94%")
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    **⚠️ Disclaimer**: This tool is for educational and research purposes only. 
    Deepfake detection is an evolving field, and no model is 100% accurate. 
    Always verify important content through multiple sources and consider the ethical implications of deepfake technology.
    """)

if __name__ == "__main__":
    main()
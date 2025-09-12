import streamlit as st
import json
import torch
import torch.nn as nn
import timm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import PIL.Image as Image
# Optional: enable HEIC/HEIF support if pillow-heif is installed
try:
    import pillow_heif  # pip install pillow-heif
    pillow_heif.register_heif_opener()
except Exception:
    pass
import streamlit.components.v1 as components
from torchvision import models as tv_models
from torchvision.models import vgg16, VGG16_Weights
import cv2

# Configure page
st.set_page_config(
    page_title="Deepfake Image Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def predict_deepfake_batch(model, device, batch_tensor, threshold: float = 0.5, temperature: float = 1.0):
    """Vectorized prediction for a batch [N,3,H,W]; returns list of result dicts."""
    with torch.no_grad():
        batch_tensor = batch_tensor.to(device)
        logits = model(batch_tensor)

        # Temperature scaling
        Tval = max(float(temperature), 1e-6)
        logits = logits / Tval

        # Convert to probabilities
        if logits.size(1) == 2:
            probs = torch.softmax(logits, dim=1)  # [N,2]
            fake_probs = probs[:, 1].cpu().numpy()
            real_probs = probs[:, 0].cpu().numpy()
        else:
            fake_probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            real_probs = 1.0 - fake_probs

        results = []
        for fp, rp in zip(fake_probs, real_probs):
            pred = "FAKE" if fp >= threshold else "REAL"
            conf = float(fp if pred == "FAKE" else rp)
            results.append({
                "prediction": pred,
                "confidence": conf,
                "fake_probability": float(fp),
                "real_probability": float(rp),
            })
        return results

# Constants
PROJECT_ROOT = Path(__file__).parent

# Temperature file is only used for MViTv2 (calibration disabled for other models)
CALIB_PATH = PROJECT_ROOT / "outputs" / "calibration" / "temperature.json"

# Multi-model registry (add more models here later)

MODEL_REGISTRY = {
    "mvitv2_tiny": {
        "display_name": "MViTv2-Tiny",
        "backend": "timm",
        "model_name": "mvitv2_tiny",
        "num_classes": 2,
        "drop_rate": 0.2,
        "checkpoint_path": PROJECT_ROOT / "outputs" / "checkpoints" / "mvitv2_phase2_best.pt",
        "input_size": 224,
        "calibration_supported": True,
    },
    "efficientnet_b3": {
        "display_name": "EfficientNet-B3",
        "backend": "torchvision",
        "model_name": "efficientnet_b3",
        "num_classes": 2,
        "drop_rate": 0.0,  # head will be provided by checkpoint; keep 0.0 here
        "checkpoint_path": PROJECT_ROOT / "B3-checkpoint" / "best_efficientnet_b3.pth",
        "input_size": 300,
        "calibration_supported": False,  # calibration only for MViTv2
    },
    "vgg16_srm": {
        "display_name": "VGG-16 + SRM",
        "backend": "custom_vgg16_srm",
        "model_name": "vgg16",
        "num_classes": 1,  # single-logit output (sigmoid)
        "drop_rate": 0.0,
        "checkpoint_path": PROJECT_ROOT / "VGG16-checkpoint" / "deepfake_cnn.pt",  # <-- you can change this path
        "input_size": 244,
        "calibration_supported": False,
    }
}

# Model performance metrics (displayed in the UI)
# Values are decimals; they will be rendered as percentages.
METRICS_REGISTRY = {
    "mvitv2_tiny": {
        "accuracy": 0.9390,
        "precision_macro": 0.9430,
        "recall_macro": 0.9393,
        "f1_macro": 0.9394,
    },
    "efficientnet_b3": {
        "accuracy": 0.9130,
        "precision_macro": 0.9200,
        "recall_macro": 0.9100,
        "f1_macro": 0.9100,
    },
    "vgg16_srm": {
        "accuracy": 0.9698,
        "precision_macro": 0.9644,
        "recall_macro": 0.9754,
        "f1_macro": 0.9699,
    },
}

# ---- Custom wrapper for VGG-16 (+ optional SRM) used by the user's notebook ----
class SRMConv(nn.Module):
    """Fixed high-pass SRM filters (3 kernels) applied depthwise to RGB -> 9ch -> 1x1 fuse -> 3ch."""
    def __init__(self):
        super().__init__()
        k = torch.tensor(
            [
                [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
                [[0, 0, 0], [0, 1, -1], [0, -1, 0]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)  # (3,1,3,3)

        # Depthwise conv: 3 input chans, 9 output chans, groups=3 -> 3 outputs per input group
        self.depthwise = nn.Conv2d(
            in_channels=3, out_channels=9, kernel_size=3, padding=1, bias=False, groups=3
        )
        with torch.no_grad():
            # Repeat the 3 kernels for each of the 3 channel groups -> (9,1,3,3)
            kernels = k.repeat(3, 1, 1, 1)
            self.depthwise.weight.copy_(kernels)
        for p in self.depthwise.parameters():
            p.requires_grad = False  # fixed SRM filters

        self.fuse = nn.Conv2d(9, 3, kernel_size=1, bias=False)  # learnable 1x1 to adapt SRM outputs

    def forward(self, x):
        x = self.depthwise(x)
        x = self.fuse(x)
        return x

class ForensicVGG16(nn.Module):
    """
    Matches the notebook: SRM -> VGG16 conv features (ImageNet features) -> GAP head (BN+Dropout+Linear) -> 1 logit.
    """
    def __init__(self, pretrained=True, dropout=0.4):
        super().__init__()
        self.srm = SRMConv()
        self.backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES if pretrained else None).features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.srm(x)
        x = self.backbone(x)
        x = self.gap(x)
        x = self.head(x)
        return x  # (N,1)

def _filter_compatible_state(model: nn.Module, state: dict) -> dict:
    """Keep only keys that exist in the model with matching tensor shapes.
    This lets us load checkpoints that may include extra keys like 'srm.*'."""
    model_state = model.state_dict()
    filtered = {}
    for k, v in state.items():
        if k in model_state and isinstance(v, torch.Tensor) and v.shape == model_state[k].shape:
            filtered[k] = v
    return filtered

@st.cache_resource(show_spinner=False)
def load_temperature(calib_path: Path) -> float:
    """Load temperature T for probability calibration; fallback to 1.0 if missing."""
    try:
        if calib_path.exists():
            data = json.loads(calib_path.read_text())
            Tval = float(data.get("temperature", 1.0))
            return max(Tval, 1e-6)
    except Exception:
        pass
    return 1.0

@st.cache_resource
def load_model(model_id: str):
    """Load the selected model according to MODEL_REGISTRY."""
    cfg = MODEL_REGISTRY[model_id]
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if cfg["backend"] == "timm":
        model = timm.create_model(
            cfg["model_name"],
            pretrained=False,
            num_classes=cfg["num_classes"],
            drop_rate=cfg.get("drop_rate", 0.0),
        )
    elif cfg["backend"] == "torchvision":
        # base backbone; we'll align the head and then load weights
        backbone = tv_models.efficientnet_b3(weights=None)
        # Torchvision EfficientNet-B3 classifier:
        # Sequential(Dropout(p=0.3), Linear(1536, 1000))
        in_features = backbone.classifier[1].in_features
        backbone.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(512, cfg["num_classes"]),
        )
        model = backbone
    elif cfg["backend"] == "custom_vgg16_srm":
        # Build the notebook-compatible VGG-16 wrapper (single-logit output)
        model = ForensicVGG16(pretrained=True, dropout=0.4)

    else:
        raise ValueError(f"Unknown backend for model_id={model_id}")

    ckpt_path = cfg["checkpoint_path"]
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("model_state")
            or checkpoint
        )
        # If the checkpoint was saved from a wrapper module (e.g., DeepfakeDetector with self.backbone),
        # its keys may be prefixed with 'backbone.'. Strip that prefix to match torchvision's module names.
        if cfg["backend"] == "torchvision" and cfg.get("model_name") == "efficientnet_b3" and isinstance(state, dict):
            if any(k.startswith("backbone.") for k in state.keys()):
                from collections import OrderedDict
                new_state = OrderedDict()
                for k, v in state.items():
                    new_key = k[9:] if k.startswith("backbone.") else k  # remove leading 'backbone.'
                    new_state[new_key] = v
                state = new_state
        # Generic prefix cleanup (DDP / wrappers)
        if isinstance(state, dict) and any(k.startswith(("module.", "model.")) for k in state.keys()):
            from collections import OrderedDict
            cleaned = OrderedDict()
            for k, v in state.items():
                if k.startswith("module."):
                    cleaned[k[len("module."):]] = v
                elif k.startswith("model."):
                    cleaned[k[len("model."):]] = v
                else:
                    cleaned[k] = v
            state = cleaned

        # For custom VGG-16 wrapper, load only matching keys (ignore SRM or other extras)
        if cfg["backend"] == "custom_vgg16_srm" and isinstance(state, dict):
            state = _filter_compatible_state(model, state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded_msg = f"Loaded {cfg['display_name']} from {ckpt_path.name}"
        if missing or unexpected:
            loaded_msg += f" (missing keys: {len(missing)}, unexpected: {len(unexpected)})"
    else:
        st.error(f"❌ Model checkpoint not found at {ckpt_path}")
        st.stop()

    model.eval()
    model.to(device)
    return model, device, cfg, loaded_msg

@st.cache_data
def get_transforms(model_id: str):
    """Get image preprocessing transforms per selected model."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_size = int(MODEL_REGISTRY[model_id]["input_size"])
    resize_to = int(round(1.10 * img_size))
    transforms = A.Compose([
        A.Resize(resize_to, resize_to, interpolation=cv2.INTER_AREA),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    return transforms, img_size

def preprocess_image(image, transforms):
    """Preprocess uploaded image for model inference."""
    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image)

    # Ensure RGB format
    if len(image.shape) == 3 and image.shape[2] == 3:
        pass  # already RGB
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]  # RGBA -> RGB
    else:
        st.error("Unsupported image format. Please upload an image with 1–4 channels.")
        return None
    
    # Apply transforms
    transformed = transforms(image=image)
    tensor_image = transformed["image"]
    
    # Add batch dimension
    tensor_image = tensor_image.unsqueeze(0)
    
    return tensor_image

def predict_deepfake(model, device, image_tensor, threshold: float = 0.5, temperature: float = 1.0):
    """Make prediction on preprocessed image with adjustable threshold."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Get model output (logits)
        logits = model(image_tensor)

        # Apply temperature scaling (divide logits by T)
        Tval = max(float(temperature), 1e-6)
        logits = logits / Tval
        
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

# Render the classification outcome (left column contents)
def render_outcome_panel(result, calib_on: bool, uncert_band: float):
    fake_prob = float(result["fake_probability"]) if isinstance(result, dict) else float(result.fake_probability)
    real_prob = float(result["real_probability"]) if isinstance(result, dict) else float(result.real_probability)
    prediction = (result["prediction"] if isinstance(result, dict) else result.prediction).upper()
    confidence = float(result["confidence"]) if isinstance(result, dict) else float(result.confidence)

    # Uncertain only when calibration is ON
    uncertain = False
    if calib_on:
        uncertain = abs(fake_prob - 0.5) <= float(uncert_band)

    # Callout
    if uncertain:
        st.info(f"⚖️ **UNCERTAIN** — {fake_prob:.1%} fake")
    else:
        if prediction == "REAL":
            st.success(f"✅  **{prediction}**")
        else:
            st.error(f"❌  **{prediction}**")

    # Confidence
    st.metric(label="Confidence", value=f"{confidence:.1%}")

    # Probabilities
    st.markdown("**Real Image Probability:**")
    st.progress(int(real_prob * 100))
    st.caption(f"{real_prob:.1%}")

    st.markdown("**Fake Image Probability:**")
    st.progress(int(fake_prob * 100))
    st.caption(f"{fake_prob:.1%}")

def render_result_card(image_pil, filename, result, calib_on: bool, uncert_band: float, idx: int, total: int, only_image: bool = False):
    """Render one image + prediction card with uncertainty handling."""
    fake_prob = result["fake_probability"]
    real_prob = result["real_probability"]
    prediction = result["prediction"]
    confidence = result["confidence"]

    # Determine uncertainty only if calibration is ON
    uncertain = False
    if calib_on:
        uncertain = abs(fake_prob - 0.5) <= float(uncert_band)

    # Center the image and caption using flexbox in markdown
    image_html = f"""
    <div style="display:flex; justify-content:center; align-items:center; flex-direction:column;">
        <img src="data:image/png;base64,{_image_to_base64(image_pil)}" style="max-width:350px; height:auto;" />
        <div style="text-align:center; color: rgba(250,250,250,0.7); margin-top:6px;">{filename}</div>
    </div>
    """
    st.markdown(image_html, unsafe_allow_html=True)

    # Carousel controls directly under the image
    ctrl_left, ctrl_center, ctrl_right = st.columns([1, 2, 1])

    def _go_prev():
        cur = int(st.session_state.get("gallery_idx", 0))
        st.session_state["gallery_idx"] = (cur - 1) % total

    def _go_next():
        cur = int(st.session_state.get("gallery_idx", 0))
        st.session_state["gallery_idx"] = (cur + 1) % total

    with ctrl_left:
        st.button("◀ Prev", use_container_width=True, key="nav_prev", on_click=_go_prev)

    with ctrl_center:
        st.markdown(
            f"<div style='text-align:center; font-weight:600;'>Image {idx + 1} of {total}</div>",
            unsafe_allow_html=True
        )

    with ctrl_right:
        st.button("Next ▶", use_container_width=True, key="nav_next", on_click=_go_next)

    # Main callout and metrics, only if not only_image
    if not only_image:
        # Main callout
        if uncertain:
            st.info(f"⚖️ **UNCERTAIN** — {fake_prob:.1%} fake")
        else:
            if prediction == "REAL":
                st.success(f"✅ **{prediction}**")
            else:
                st.error(f"❌ **{prediction}**")

        # Confidence
        st.metric(label="Confidence", value=f"{confidence:.1%}")

        # Probabilities
        st.markdown("**Real Image Probability:**")
        st.progress(int(real_prob * 100))
        st.caption(f"{real_prob:.1%}")

        st.markdown("**Fake Image Probability:**")
        st.progress(int(fake_prob * 100))
        st.caption(f"{fake_prob:.1%}")

def _image_to_base64(image_pil):
    import io
    import base64
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def main():
    # Title and description
    st.title("🔍 Deepfake Image Detection")
    st.markdown("""
    Upload one or more images and choose a model to classify them as **Real** or **Fake**.
    """)
    
    # Sidebar
    with st.sidebar:
        # Model selector
        model_options = {MODEL_REGISTRY[k]["display_name"]: k for k in MODEL_REGISTRY.keys()}
        selected_label = st.selectbox("Model", list(model_options.keys()), index=0)
        model_id = model_options[selected_label]

        # Load model & transforms based on selection
        model, device, model_cfg, loaded_msg = load_model(model_id)
        st.session_state["loaded_msg"] = loaded_msg
        transforms, input_size = get_transforms(model_id)

        # Title and loaded model info
        st.title("Deepfake Detector")
        if st.session_state.get("loaded_msg"):
            st.caption(st.session_state["loaded_msg"])

        threshold = st.slider("Decision threshold (fake if ≥ threshold)", min_value=0.30, max_value=0.90, value=0.50, step=0.01)
        
        # Calibration controls (only for MViTv2)
        calib_supported = bool(model_cfg.get("calibration_supported", False))
        if calib_supported:
            calib_on = st.toggle("Enable calibration (temperature scaling)", value=False, help="When ON, apply the learned temperature to logits before softmax.")
            uncert_band = st.slider("Uncertain zone (± around 50%)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, help="Only used when calibration is ON. Samples within [0.5±band] are flagged as 'Uncertain'.")
            temperature = load_temperature(CALIB_PATH) if calib_on else 1.0
        else:
            calib_on = False
            uncert_band = 0.0
            temperature = 1.0
            st.caption("Calibration not available for this model.")
    
    # --- Single-uploader state (we keep files once chosen) ---
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    # Add instructions and notes columns above uploader
    help_left, help_right = st.columns(2)
    with help_left:
        st.subheader("📋 Instructions")
        st.markdown(
            """
            1. Upload one or more face images (JPG/PNG/JPEG)
            2. The model analyzes each image
            3. See **prediction** and **confidence** below
            4. Green = Real, Red = Fake
            """
        )
    with help_right:
        st.subheader("⚠️ Notes")
        st.markdown(
            """
            - Works best with clear, frontal faces
            - Trained on a specific dataset
            - For research/education purposes
            - Not 100% accurate — use with caution
            """
        )

    # Always show the uploader (top). Keeps existing files in session; adding new files replaces the list.
    st.header("📤 Upload Image(s)")
    files = st.file_uploader(
        "Choose image file(s)",
        type=["png", "jpg", "jpeg", "webp", "bmp", "heic", "heif"],
        accept_multiple_files=True,
        help="Upload one or more face images to analyze",
        label_visibility="collapsed",
        key="uploader_main"
    )
    if files is not None and len(files) > 0:
        st.session_state["uploaded_files"] = files
        st.session_state["just_uploaded"] = True  # trigger smooth scroll once on next render
            

    if st.session_state["uploaded_files"]:
        uploaded_files = st.session_state["uploaded_files"]

        # Ensure the results anchor exists BEFORE attempting to scroll
        st.markdown("<div id='results-anchor'></div>", unsafe_allow_html=True)

        # If we just uploaded files, auto-scroll to the results anchor (no global rerun needed)
        if st.session_state.get("just_uploaded") and not st.session_state.get("auto_scrolled", False):
            # Consume the scroll intent now so it doesn't leak into the next rerun (e.g., button clicks)
            st.session_state["auto_scrolled"] = True
            st.session_state["just_uploaded"] = False

            components.html(
                """
                <style>
                /* Ensure zero margins inside the helper iframe */
                html, body { margin: 0 !important; padding: 0 !important; height: 0 !important; overflow: hidden !important; }
                </style>
                <script>
                (function(){
                  try {
                    const frame = window.frameElement;
                    // Immediately collapse the iframe to avoid any visible gap
                    if (frame) {
                      frame.style.height = '0px';
                      frame.style.minHeight = '0px';
                      frame.style.border = '0';
                    }
                    const collapseParent = () => {
                      if (!frame) return;
                      const parent = frame.parentElement;
                      if (parent) {
                        parent.style.height = '0px';
                        parent.style.minHeight = '0px';
                        parent.style.margin = '0';
                        parent.style.padding = '0';
                        parent.style.overflow = 'hidden';
                      }
                    };
                    // Perform the smooth scroll after a short delay, then hide the frame entirely
                    setTimeout(() => {
                      try {
                        const anchor = parent.document.querySelector('#results-anchor');
                        if (anchor) { anchor.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
                      } catch (_) {}
                      // Fully hide after scroll just in case
                      if (frame) frame.style.display = 'none';
                      collapseParent();
                    }, 1500);
                  } catch (e) {
                    // no-op
                  }
                })();
                </script>
                """,
                height=0
            )

        # Compute effective temperature for calibration
        T_effective = float(temperature) if calib_on else 1.0

        # Lightweight preview first: load images + preprocess tensors
        images, tensors = [], []
        for f in uploaded_files:
            try:
                img_pil = Image.open(f).convert("RGB")
            except Exception as e:
                name = getattr(f, "name", "image")
                # Common case: HEIC/HEIF uploaded but pillow-heif not installed
                if str(name).lower().endswith((".heic", ".heif")):
                    st.error("Could not open HEIC/HEIF file. Please install the optional dependency: `pip install pillow-heif` and restart the app.")
                else:
                    st.error(f"Could not open {name}: {e}")
                continue

            tensor = preprocess_image(img_pil, transforms)
            if tensor is None:
                st.error("Failed to preprocess image")
                continue

            images.append(img_pil)
            tensors.append(tensor)

        # Batch predict (vectorized) with a local spinner so only results area shows busy state
        results = []
        if len(tensors) > 0:
            batch = torch.cat(tensors, dim=0)  # [N,3,224,224]
            with st.spinner(f"Analyzing {len(tensors)} image(s)..."):
                # (Optional) chunk if very large to save memory
                MAX_BATCH = 64
                if batch.size(0) <= MAX_BATCH:
                    results = predict_deepfake_batch(
                        model, device, batch, threshold=threshold, temperature=T_effective
                    )
                else:
                    results = []
                    for i in range(0, batch.size(0), MAX_BATCH):
                        chunk = batch[i:i+MAX_BATCH]
                        results.extend(
                            predict_deepfake_batch(
                                model, device, chunk, threshold=threshold, temperature=T_effective
                            )
                        )

        # Results section (anchor already placed above)
        st.header("🎯 Detection Result")
        st.write("")
        if len(images) == 0:
            st.info("No valid images to display.")
        else:
            # Carousel state
            n = len(images)
            if "gallery_idx" not in st.session_state or st.session_state.get("gallery_count") != n:
                st.session_state["gallery_idx"] = 0
                st.session_state["gallery_count"] = n
            idx = st.session_state["gallery_idx"]

            render_result_card(
                images[idx], getattr(uploaded_files[idx], 'name', f'image_{idx}'),
                results[idx],
                calib_on=bool(calib_on),
                uncert_band=float(uncert_band),
                idx=idx,
                total=n,
                only_image=True,
            )

            # Side-by-side: left = outcome, right = technical details
            col_outcome, col_tech = st.columns([2, 1])
            with col_outcome:
                render_outcome_panel(
                    results[idx],
                    calib_on=bool(calib_on),
                    uncert_band=float(uncert_band),
                )
            with col_tech:
                st.subheader("🔬 Technical Details")
                st.markdown(f"""
                - **Model**: {model_cfg['display_name']}
                - **Device**: {device.type.upper()}
                - **Input Shape**: {[1, 3, input_size, input_size]}
                - **Preprocessing**: Resize to {input_size}×{input_size}, ImageNet normalization
                - **Decision Threshold**: {threshold:.2f}
                - **Calibration**: {"ON" if calib_on else "OFF"}
                - **Temperature (T)**: {float(temperature):.3f}
                - **Uncertain Zone**: ±{float(uncert_band):.2f} (around 50% when calibration is ON)
                """)


    else:
        st.info("👆 Upload image(s) to see detection results")
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
    metrics = METRICS_REGISTRY.get(model_id)
    if metrics is not None:
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        with col2:
            st.metric("Precision (Macro)", f"{metrics['precision_macro']*100:.2f}%")
        with col3:
            st.metric("Recall (Macro)", f"{metrics['recall_macro']*100:.2f}%")
        with col4:
            st.metric("F1 (Macro)", f"{metrics['f1_macro']*100:.2f}%")
    else:
        with col1:
            st.metric("Accuracy", "-")
        with col2:
            st.metric("Precision (Macro)", "-")
        with col3:
            st.metric("Recall (Macro)", "-")
        with col4:
            st.metric("F1 (Macro)", "-")

    # Disclaimer
    st.markdown("---")
    st.markdown("""
    **⚠️ Disclaimer**: This tool is for educational and research purposes only. 
    Deepfake detection is an evolving field, and no model is 100% accurate. 
    Always verify important content through multiple sources and consider the ethical implications of deepfake technology.
    """)

if __name__ == "__main__":
    main()
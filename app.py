import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import os

# ----------------- Configuration & Subtype Mapping -----------------
st.set_page_config(
    page_title="BreaKHis Subtype Classifier",
    page_icon="🔬",
    layout="wide"
)

SUBTYPE_FULL_NAMES = {
    'A': 'Adenosis (Benign)',
    'F': 'Fibroadenoma (Benign)',
    'PT': 'Phyllodes Tumor (Benign)',
    'TA': 'Tubular Adenoma (Benign)',
    'DC': 'Ductal Carcinoma (Malignant)',
    'LC': 'Lobular Carcinoma (Malignant)',
    'MC': 'Mucinous Carcinoma (Malignant)',
    'PC': 'Papillary Carcinoma (Malignant)',
}

IMG_H, IMG_W = 384, 576
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)


# ----------------- Model Architecture -----------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
    def forward(self, x):
        b, c = x.shape[:2]
        a = self.mlp(x.mean((2, 3))) + self.mlp(x.amax((2, 3)))
        return x * (2 * torch.sigmoid(a)).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
    def forward(self, x):
        a = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * (2 * torch.sigmoid(self.conv(a)))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps
    def forward(self, x):
        x = x.float().clamp(min=self.eps)
        return x.pow(self.p).mean((2, 3)).pow(1.0 / self.p)

class MultiScaleCBAMNet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        f = models.densenet121(weights=None).features
        self.stem = nn.Sequential(
            f.conv0, f.norm0, f.relu0, f.pool0,
            f.denseblock1, f.transition1, f.denseblock2
        )
        self.t2_norm = f.transition2.norm
        self.t2_rest = nn.Sequential(f.transition2.conv, f.transition2.pool)
        self.block3 = f.denseblock3
        self.t3_norm = f.transition3.norm
        self.t3_rest = nn.Sequential(f.transition3.conv, f.transition3.pool)
        self.block4 = f.denseblock4
        self.norm5 = f.norm5

        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(1024)
        self.pool = GeM()
        self.pool2 = GeM()
        self.pool3 = GeM()
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512 + 1024 + 1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        p2 = F.relu(self.t2_norm(x))
        x = self.block3(self.t2_rest(p2))
        p3 = F.relu(self.t3_norm(x))
        x = self.block4(self.t3_rest(p3))
        p4 = F.relu(self.norm5(x))
        z = torch.cat([
            self.pool2(self.cbam2(p2)),
            self.pool3(self.cbam3(p3)),
            self.pool(self.cbam4(p4))
        ], dim=1)
        return self.head(z)


# ----------------- Image Preprocessing -----------------
def reinhard_normalize(source_rgb, target_rgb):
    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_mean, src_std = src_lab.mean(axis=(0, 1)), src_lab.std(axis=(0, 1)) + 1e-6
    tgt_mean, tgt_std = tgt_lab.mean(axis=(0, 1)), tgt_lab.std(axis=(0, 1))
    result = (src_lab - src_mean) * (tgt_std / src_std) + tgt_mean
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)


@st.cache_resource
def load_trained_model(weight_path):
    checkpoint = torch.load(weight_path, map_location=DEVICE)
    classes = checkpoint.get('classes', ['A', 'DC', 'F', 'LC', 'MC', 'PC', 'PT', 'TA'])
    model = MultiScaleCBAMNet(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, classes


def predict(image_rgb, model, classes, use_tta=True, ref_target=None):
    if ref_target is not None:
        processed_rgb = reinhard_normalize(image_rgb, ref_target)
    else:
        processed_rgb = image_rgb

    resized = cv2.resize(processed_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(DEVICE)

    views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)] if use_tta else [x]
    probs = 0.0
    with torch.no_grad():
        for v in views:
            v_norm = (v - NORM_MEAN) / NORM_STD
            logits = model(v_norm)
            probs += torch.softmax(logits, dim=1)
        probs = (probs / len(views)).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    return classes[top_idx], probs


# ----------------- UI Dashboard -----------------
st.title("🔬 Histopathology Breast Cancer Subtype Classifier")
st.markdown("Multi-Scale CBAM DenseNet-121 inference on BreaKHis slide images.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Setup")
    mag_choice = st.selectbox("Select Magnification", ["40X", "100X", "200X", "400X"], index=0)
    use_tta = st.checkbox("Enable 4-View TTA (Flips)", value=True)
    
    weight_path = f"models/multiscale_cbam_{mag_choice}.pth"
    if not os.path.exists(weight_path):
        st.error(f"Checkpoint `{weight_path}` not found. Please upload it to the `models/` directory.")
        st.stop()
    
    model, classes = load_trained_model(weight_path)
    
    # Optional reference image for Reinhard normalization
    ref_path = "assets/reference_stain.png"
    ref_target = None
    if os.path.exists(ref_path):
        ref_bgr = cv2.imread(ref_path)
        if ref_bgr is not None:
            ref_target = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

    uploaded_file = st.file_uploader("Upload Histopathology Image (PNG / JPG)", type=["png", "jpg", "jpeg"])

with col2:
    st.subheader("2. Analysis & Results")
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(pil_image)
        st.image(pil_image, caption="Uploaded Slide Image", use_column_width=True)

        with st.spinner("Analyzing slide features..."):
            pred_class, probs = predict(image_rgb, model, classes, use_tta=use_tta, ref_target=ref_target)
            full_diagnosis = SUBTYPE_FULL_NAMES.get(pred_class, pred_class)
            confidence = probs[classes.index(pred_class)] * 100

        is_malignant = "(Malignant)" in full_diagnosis
        alert_fn = st.error if is_malignant else st.success
        alert_fn(f"**Predicted Subtype:** {full_diagnosis} — **{confidence:.2f}%** Confidence")

        st.markdown("#### Class Probabilities")
        prob_dict = {SUBTYPE_FULL_NAMES.get(c, c): float(probs[i]) for i, c in enumerate(classes)}
        st.bar_chart(prob_dict)
    else:
        st.info("Upload a microscope image on the left panel to run inference.")
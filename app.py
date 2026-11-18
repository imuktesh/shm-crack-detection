
import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load models ----------
model_cls = models.resnet18()
model_cls.fc = nn.Linear(model_cls.fc.in_features, 2)
model_cls.load_state_dict(torch.load("best_resnet18_sdnet.pth", map_location=device, weights_only=True))
model_cls = model_cls.to(device).eval()

model_seg = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
model_seg.load_state_dict(torch.load("best_unet_v2.pth", map_location=device, weights_only=True))
model_seg = model_seg.to(device).eval()

cls_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ---------- Helpers ----------
def to_seg_tensor(img_rgb, size=512):
    img = cv2.resize(img_rgb, (size, size)).astype(np.float32) / 255.0
    img = (img - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
    return torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(device)

def predict_with_tta(model, x):
    with torch.no_grad():
        p1 = torch.sigmoid(model(x))
        p2 = torch.flip(torch.sigmoid(model(torch.flip(x, dims=[3]))), dims=[3])
        p3 = torch.flip(torch.sigmoid(model(torch.flip(x, dims=[2]))), dims=[2])
        p4 = torch.flip(torch.sigmoid(model(torch.flip(x, dims=[2,3]))), dims=[2,3])
    return (p1+p2+p3+p4)/4

def postprocess_mask(m, min_area=20):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out

def _junctions_endpoints(skel):
    s = skel.astype(np.uint8)
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nbr = cv2.filter2D(s, -1, kernel)
    return int(((s==1)&(nbr>=3)).sum()), int(((s==1)&(nbr==1)).sum())

def _orientation(skel):
    ys, xs = np.where(skel)
    if len(xs) < 5: return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    pts -= pts.mean(axis=0)
    val, vec = np.linalg.eigh(np.cov(pts.T))
    p = vec[:, np.argmax(val)]
    return float(abs(np.degrees(np.arctan2(p[1], p[0]))) % 180)

def _orient_score(a):
    a = a if a <= 90 else 180 - a
    return max(0.0, 1.0 - abs(a - 45)/45.0)

def structural_analysis(mask, mm_per_px):
    if mask.sum() == 0: return None
    skel = skeletonize(mask > 0)
    if skel.sum() < 5: return None

    dist = distance_transform_edt(mask)
    widths = (dist[skel]*2) - 1
    widths = widths[widths > 0]
    if len(widths) == 0: return None

    max_w  = float(widths.max())  * mm_per_px
    avg_w  = float(widths.mean()) * mm_per_px
    angle  = _orientation(skel)
    junc, ends = _junctions_endpoints(skel)
    length = int(skel.sum())

    o_s = _orient_score(angle)
    b_s = min(junc/5.0, 1.0)
    l_s = min(length/500.0, 1.0)
    p   = 0.5*o_s + 0.3*b_s + 0.2*l_s

    if max_w < 0.3:    sev = "LOW"
    elif max_w <= 1.0: sev = "MEDIUM"
    else:              sev = "HIGH"

    if p > 0.65:    rec, col = "STRUCTURAL — repair recommended", "red"
    elif p > 0.40:  rec, col = "POSSIBLY STRUCTURAL — monitor", "orange"
    else:           rec, col = "LIKELY NON-STRUCTURAL (cosmetic)", "green"

    return dict(max_w=max_w, avg_w=avg_w, angle=angle, junc=junc,
                length=length, p=p, sev=sev, rec=rec, col=col, skel=skel)

# ---------- Pipeline ----------
def pipeline(input_image, mm_per_px):
    if input_image is None:
        return None, "Please upload an image."

    if len(input_image.shape) == 2:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)

    img = input_image.copy()
    H, W = img.shape[:2]

    # Stage 1: classify
    cls_t = cls_tf(Image.fromarray(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model_cls(cls_t), dim=1)[0, 1].item()

    if prob < 0.40:
        return img, (
            f"✅ NO CRACK DETECTED\n\n"
            f"Classification confidence: {(1-prob)*100:.2f}%\n"
            f"Structure appears safe."
        )

    # Stage 2: segment with TTA
    seg_t = to_seg_tensor(img, 512)
    mask_512 = (predict_with_tta(model_seg, seg_t) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    mask_512 = postprocess_mask(mask_512)
    mask = cv2.resize(mask_512, (W, H), interpolation=cv2.INTER_NEAREST)

    # Stage 3: structural analysis
    res = structural_analysis(mask, mm_per_px)

    overlay = img.copy()
    if res is not None:
        overlay[res["skel"]] = [255, 0, 0]
        report = (
            f"⚠️ CRACK DETECTED\n"
            f"Crack probability: {prob*100:.2f}%\n\n"
            f"━━━━ MEASUREMENTS ━━━━\n"
            f"Max width: {res['max_w']:.2f} mm\n"
            f"Avg width: {res['avg_w']:.2f} mm\n"
            f"Orientation: {res['angle']:.1f}°\n"
            f"Length: {res['length']} px\n"
            f"Junctions: {res['junc']}\n\n"
            f"━━━━ ASSESSMENT ━━━━\n"
            f"Severity (ACI 224R): {res['sev']}\n"
            f"Structural score: {res['p']*100:.1f}%\n\n"
            f"→ {res['rec']}"
        )
    else:
        report = f"⚠️ Crack detected ({prob*100:.2f}%) but mask too thin to measure."

    return overlay, report

# ---------- UI ----------
demo = gr.Interface(
    fn=pipeline,
    inputs=[
        gr.Image(label="Upload building photo", type="numpy"),
        gr.Slider(0.01, 1.0, value=0.05, step=0.01,
                  label="Scale (mm per pixel)",
                  info="Close-up: 0.02-0.05 | Mid-range: 0.1-0.2 | Wide: 0.3+"),
    ],
    outputs=[
        gr.Image(label="Crack centreline (red)"),
        gr.Textbox(label="SHM report", lines=16),
    ],
    title="🏗️ AI Structural Health Monitoring",
    description=(
        "Two-stage pipeline: crack detection → segmentation → structural assessment. "
        "Severity thresholds per ACI 224R. Structural score is a morphology-based "
        "triage indicator, not a substitute for on-site inspection."
    ),
    examples=[],
)

if __name__ == "__main__":
    demo.launch()

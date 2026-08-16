import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel
from torchvision import models as tv_models

from image_preprocessing import apply_crop_mode, build_inference_transform, detect_face_crop
from models import SimpleCNN


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_COPY = {
    "angry": "Tighter brows and compressed lips are pushing the model toward a high-tension read.",
    "disgust": "The model is picking up facial cues that usually align with aversion or discomfort.",
    "fear": "Wide-eye tension and guarded expression signals are making fear the strongest match.",
    "happy": "The strongest cues suggest ease, warmth, or a smile-driven expression.",
    "neutral": "The face reads as comparatively balanced, with fewer strong emotion-specific cues.",
    "sad": "Lower facial energy and downward tension are pulling the result toward sadness.",
    "surprise": "Open-eye and open-mouth signals are making surprise the clearest match.",
}
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "resnet18")
DEFAULT_WEIGHTS_PATH = Path(os.getenv("MODEL_WEIGHTS", "best_resnet18.pt"))
DEFAULT_CROP_MODE = os.getenv("CROP_MODE", "face")

app = FastAPI(title="FaceImp Emotion Studio")


class PredictRequest(BaseModel):
    image: str
    crop_mode: str = DEFAULT_CROP_MODE
    top_k: int = 3
    browser_crop_strategy: Optional[str] = None


# Use the best available accelerator automatically, while still supporting plain CPU machines.
def resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Decode a browser data URL so webcam captures and uploaded images share the same server-side path.
def decode_data_url(data_url: str) -> Image.Image:
    if "," not in data_url:
        raise ValueError("Expected a data URL like data:image/jpeg;base64,...")

    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# Convert a PIL image into a browser-safe data URL so the UI can display exactly what the model saw.
def encode_pil_to_data_url(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{image_format.lower()};base64,{encoded}"


# Keep the same face-first preprocessing logic used by the custom-image scripts.
def prepare_image(image: Image.Image, crop_mode: str):
    face_detected = False
    processed = image

    if crop_mode == "face":
        detected = detect_face_crop(image)
        if detected is not None:
            processed = detected
            face_detected = True
        else:
            processed = apply_crop_mode(image, "tight")
    else:
        processed = apply_crop_mode(image, crop_mode)

    return processed, face_detected


# Undo normalization so the browser can show a human-readable preview of the model input.
def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().clone()
    if image.shape[0] == 1:
        image = image * 0.5 + 0.5
        image = image.clamp(0, 1).squeeze(0).numpy()
        return np.stack([image, image, image], axis=-1)

    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    image = image * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


# Use a small NumPy-based colormap so the deployment does not need a heavier plotting stack.
def apply_turbo_colormap(cam_map: np.ndarray) -> np.ndarray:
    cam = np.clip(cam_map, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


# Blend the normalized Grad-CAM map onto the model input image so the result is easy to interpret in the UI.
def build_overlay_image(display_image: np.ndarray, cam_map: np.ndarray) -> Image.Image:
    heat = apply_turbo_colormap(cam_map)
    overlay = np.clip(0.58 * display_image + 0.42 * heat, 0, 1)
    return Image.fromarray((overlay * 255).astype("uint8"))


class WebEmotionRuntime:
    """Load the trained model once and reuse it for browser predictions and Grad-CAM."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, weights_path: Path = DEFAULT_WEIGHTS_PATH):
        self.model_name = model_name
        self.weights_path = Path(weights_path)
        self.device = resolve_device()
        self.activations = None
        self.gradients = None
        self.model, self.transform, self.target_layer = self._load_model_and_transform()
        self._register_hooks()

    def _load_model_and_transform(self):
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Could not find checkpoint: {self.weights_path}. Place the trained .pt file next to "
                "web_app.py or set the MODEL_WEIGHTS environment variable."
            )

        if self.model_name == "cnn":
            model = SimpleCNN()
            target_layer = model.features[-2]
        else:
            model = tv_models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
            target_layer = model.layer4[-1].conv2

        model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        transform = build_inference_transform(self.model_name)
        return model, transform, target_layer

    # Hooks capture the feature maps and gradients required to compute Grad-CAM for the predicted class.
    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def predict_with_gradcam(
        self,
        image: Image.Image,
        crop_mode: str = DEFAULT_CROP_MODE,
        top_k: int = 3,
        browser_crop_strategy: Optional[str] = None,
    ):
        processed_image, face_detected = prepare_image(image, crop_mode)
        tensor = self.transform(processed_image).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, class_index = probabilities.max(dim=0)
        top_count = min(max(top_k, 1), len(CLASS_NAMES))
        top_values, top_indices = probabilities.topk(top_count)

        score = logits[0, class_index]
        self.model.zero_grad(set_to_none=True)
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)

        cam_map = cam[0, 0]
        cam_map = cam_map - cam_map.min()
        cam_map = cam_map / (cam_map.max() + 1e-8)
        cam_map = cam_map.detach().cpu().numpy()

        display_image = denormalize_tensor(tensor[0])
        model_input_image = Image.fromarray((display_image * 255).astype("uint8"))
        overlay_image = build_overlay_image(display_image, cam_map)

        return {
            "predicted_emotion": CLASS_NAMES[int(class_index.item())],
            "confidence": float(confidence.item()),
            "emotion_copy": EMOTION_COPY[CLASS_NAMES[int(class_index.item())]],
            "top_predictions": [
                {
                    "emotion": CLASS_NAMES[int(index.item())],
                    "confidence": float(value.item()),
                }
                for value, index in zip(top_values, top_indices)
            ],
            "crop_mode": crop_mode,
            "face_detected": face_detected,
            "browser_crop_strategy": browser_crop_strategy,
            "model": self.model_name,
            "device": self.device,
            "model_input_image": encode_pil_to_data_url(model_input_image),
            "grad_cam_overlay": encode_pil_to_data_url(overlay_image),
        }


# Load the checkpoint once per process so browser predictions stay responsive.
@lru_cache(maxsize=1)
def load_runtime(model_name: str = DEFAULT_MODEL_NAME, weights_path: str = str(DEFAULT_WEIGHTS_PATH)):
    return WebEmotionRuntime(model_name=model_name, weights_path=Path(weights_path))


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FaceImp Emotion Studio</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
      --paper: #f7efe2;
      --paper-2: #fef8ef;
      --panel: rgba(255, 251, 244, 0.88);
      --glass: rgba(255, 255, 255, 0.62);
      --ink: #16352d;
      --muted: #6e7f78;
      --line: rgba(22, 53, 45, 0.12);
      --accent: #e56f4a;
      --accent-soft: #efb86f;
      --mint: #d8e6dd;
      --mint-strong: #24493f;
      --warn: #bb6a40;
      --shadow: 0 26px 70px rgba(24, 43, 36, 0.12);
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: 'Manrope', sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255, 250, 239, 0.95), transparent 30%),
        radial-gradient(circle at top right, rgba(221, 234, 224, 0.88), transparent 34%),
        linear-gradient(160deg, #f7efe2 0%, #f3ecdf 36%, #e8efe8 100%);
      padding: 24px 18px 42px;
    }
    h1, h2, h3, h4 {
      font-family: 'Fraunces', serif;
      letter-spacing: -0.03em;
      margin: 0;
    }
    p { margin: 0; color: var(--muted); line-height: 1.65; }
    .page {
      max-width: 1260px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }
    .hero {
      border-radius: 30px;
      padding: 30px;
      background: linear-gradient(140deg, rgba(255,255,255,0.74), rgba(255,247,237,0.70));
      border: 1px solid rgba(255,255,255,0.72);
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(22, 53, 45, 0.07);
      color: var(--ink);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .hero h1 {
      margin-top: 14px;
      font-size: clamp(2.3rem, 4vw, 4rem);
      line-height: 0.98;
      max-width: 820px;
    }
    .hero p {
      margin-top: 14px;
      max-width: 760px;
      font-size: 1.02rem;
    }
    .hero-stats {
      margin-top: 24px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }
    .hero-stat {
      border-radius: 22px;
      padding: 16px 18px;
      background: rgba(255,255,255,0.7);
      border: 1px solid var(--line);
    }
    .hero-stat strong {
      display: block;
      margin-top: 4px;
      font-size: 1.25rem;
      color: var(--ink);
    }
    .hero-stat span {
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.08em;
      color: var(--muted);
      text-transform: uppercase;
    }
    .shell {
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 20px;
      align-items: start;
    }
    .panel {
      border-radius: 28px;
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.74);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
      padding: 22px;
    }
    .section-heading {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: end;
      margin-bottom: 14px;
    }
    .section-heading p {
      font-size: 0.95rem;
      max-width: 560px;
    }
    .step-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .step-card {
      border-radius: 20px;
      padding: 16px;
      background: rgba(255,255,255,0.68);
      border: 1px solid var(--line);
    }
    .step-card span {
      display: inline-flex;
      width: 34px;
      height: 34px;
      align-items: center;
      justify-content: center;
      border-radius: 50%;
      background: rgba(229,111,74,0.12);
      color: var(--accent);
      font-weight: 800;
      margin-bottom: 12px;
    }
    .step-card strong {
      display: block;
      color: var(--ink);
      margin-bottom: 6px;
    }
    .frame {
      position: relative;
      width: 100%;
      aspect-ratio: 4 / 3;
      border-radius: 24px;
      overflow: hidden;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at top, rgba(255,255,255,0.08), transparent 30%),
        linear-gradient(180deg, #1d302a, #0d1614);
      border: 1px solid rgba(36, 73, 63, 0.18);
      isolation: isolate;
    }
    video, img, canvas {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    canvas { display: none; }
    .empty-state {
      position: absolute;
      inset: 0;
      padding: 26px;
      display: grid;
      place-items: center;
      text-align: center;
      background:
        radial-gradient(circle at 50% 18%, rgba(239, 184, 111, 0.20), transparent 32%),
        linear-gradient(180deg, rgba(14, 26, 23, 0.80), rgba(10, 18, 15, 0.94));
      color: rgba(245, 240, 232, 0.80);
    }
    .empty-state strong {
      display: block;
      margin-bottom: 8px;
      color: #fff8ef;
      font-size: 1.15rem;
    }
    .control-stack {
      display: grid;
      gap: 16px;
      margin-top: 18px;
    }
    .control-row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }
    .field-group {
      display: grid;
      gap: 8px;
      min-width: 210px;
      flex: 1 1 220px;
    }
    .field-group label {
      font-size: 12px;
      font-weight: 800;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    select, button, .upload-chip {
      border-radius: 999px;
      font: inherit;
      min-height: 48px;
      padding: 0 18px;
      transition: transform 180ms ease, box-shadow 180ms ease, opacity 180ms ease;
    }
    select {
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.7);
      color: var(--ink);
      outline: none;
    }
    button, .upload-chip {
      border: none;
      cursor: pointer;
      font-weight: 800;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      text-decoration: none;
    }
    button:hover, .upload-chip:hover { transform: translateY(-1px); }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.5;
      transform: none;
      box-shadow: none;
    }
    .primary {
      background: linear-gradient(135deg, var(--accent), var(--accent-soft));
      color: white;
      box-shadow: 0 20px 34px rgba(229,111,74,0.24);
    }
    .secondary {
      background: rgba(22,53,45,0.08);
      color: var(--ink);
      border: 1px solid rgba(22,53,45,0.08);
    }
    .upload-chip {
      background: rgba(36, 73, 63, 0.92);
      color: #fdf7ef;
      box-shadow: 0 18px 28px rgba(36,73,63,0.18);
    }
    .status {
      min-height: 1.6em;
      font-size: 0.97rem;
      font-weight: 700;
      color: var(--mint-strong);
    }
    .warning {
      color: var(--warn);
    }
    .capture-note {
      font-size: 0.94rem;
      color: var(--muted);
      margin-top: 12px;
    }
    .summary-card {
      border-radius: 28px;
      padding: 24px;
      background: linear-gradient(155deg, rgba(22,53,45,0.97), rgba(34,73,63,0.92));
      color: #fbf6ee;
      box-shadow: 0 30px 72px rgba(22, 53, 45, 0.22);
    }
    .summary-label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 800;
      color: rgba(251,246,238,0.72);
    }
    .summary-emotion {
      margin-top: 12px;
      font-size: clamp(2rem, 3vw, 3.2rem);
      line-height: 0.95;
    }
    .summary-confidence {
      margin-top: 10px;
      color: #f0c27d;
      font-size: 1.05rem;
      font-weight: 800;
    }
    .summary-copy {
      margin-top: 16px;
      color: rgba(251,246,238,0.88);
      line-height: 1.7;
      font-size: 0.98rem;
    }
    .meta-grid {
      margin-top: 18px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .meta-chip {
      border-radius: 18px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.06);
    }
    .meta-chip span {
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: rgba(251,246,238,0.66);
      font-weight: 800;
    }
    .meta-chip strong {
      display: block;
      margin-top: 4px;
      font-size: 0.95rem;
      color: #fff8ef;
    }
    .candidate-panel, .preview-card {
      margin-top: 18px;
      border-radius: 22px;
      padding: 18px;
      background: rgba(255,255,255,0.7);
      border: 1px solid var(--line);
    }
    .candidate-list {
      display: grid;
      gap: 12px;
      margin-top: 14px;
    }
    .candidate-row {
      border-radius: 16px;
      padding: 14px 14px 12px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(22,53,45,0.08);
    }
    .candidate-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
    }
    .candidate-head strong {
      text-transform: capitalize;
      color: var(--ink);
    }
    .candidate-head span {
      color: var(--muted);
      font-weight: 700;
    }
    .candidate-bar {
      margin-top: 10px;
      height: 9px;
      border-radius: 999px;
      background: rgba(22,53,45,0.08);
      overflow: hidden;
    }
    .candidate-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent), var(--accent-soft));
    }
    .preview-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 18px;
    }
    .preview-card h3 { margin-bottom: 6px; }
    .preview-card p { font-size: 0.92rem; }
    .preview-card img {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      border-radius: 16px;
      border: 1px solid rgba(22,53,45,0.12);
      margin-top: 12px;
      background: rgba(16, 26, 23, 0.92);
    }
    .preview-placeholder {
      margin-top: 12px;
      aspect-ratio: 1 / 1;
      border-radius: 16px;
      border: 1px dashed rgba(22,53,45,0.16);
      display: grid;
      place-items: center;
      text-align: center;
      padding: 18px;
      color: var(--muted);
      background: rgba(255,255,255,0.44);
    }
    .pill {
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(229,111,74,0.12);
      color: var(--accent);
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-top: 4px;
    }
    .utility-note {
      margin-top: 16px;
      border-radius: 18px;
      padding: 14px 16px;
      background: rgba(36,73,63,0.08);
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.6;
    }
    @media (max-width: 980px) {
      .shell { grid-template-columns: 1fr; }
      .hero-stats, .step-grid, .preview-grid, .meta-grid { grid-template-columns: 1fr; }
      .hero { padding: 24px; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <span class="eyebrow">FaceImp Emotion Studio</span>
      <h1>A more polished way to explore what the emotion model is seeing.</h1>
      <p>
        This live demo keeps the same trained checkpoint and Grad-CAM logic from the project, but wraps it in a cleaner experience:
        capture a snapshot or upload an image, preview the exact crop sent to the model, and compare the strongest candidate emotions in a way that feels much more intentional.
      </p>
      <div class="hero-stats">
        <div class="hero-stat"><span>Workflow</span><strong>Webcam + uploads</strong></div>
        <div class="hero-stat"><span>Explainability</span><strong>Grad-CAM overlay</strong></div>
        <div class="hero-stat"><span>Model</span><strong>ResNet18 checkpoint</strong></div>
      </div>
    </section>

    <div class="shell">
      <section class="panel">
        <div class="section-heading">
          <div>
            <h2>Capture and confirm</h2>
            <p>Use the webcam or upload a portrait. The app keeps the exact still image visible until you confirm that this is the frame you want the model to analyze.</p>
          </div>
        </div>

        <div class="step-grid">
          <div class="step-card"><span>1</span><strong>Open or upload</strong><p>Start the camera or choose a file to create the prediction candidate image.</p></div>
          <div class="step-card"><span>2</span><strong>Review framing</strong><p>Retake if needed. The large frame acts as your confirmation preview before any prediction is made.</p></div>
          <div class="step-card"><span>3</span><strong>Analyze snapshot</strong><p>Once confirmed, the app sends that still image through the same emotion pipeline used by the project scripts.</p></div>
        </div>

        <div class="frame">
          <video id="video" autoplay playsinline muted hidden></video>
          <img id="snapshot" alt="Captured snapshot" hidden />
          <canvas id="canvas"></canvas>
          <div id="framePlaceholder" class="empty-state">
            <div>
              <strong>No image selected yet</strong>
              Start the camera for a live preview or upload a portrait. After you take or choose an image, that exact still will remain here until you confirm the prediction.
            </div>
          </div>
        </div>
        <p class="capture-note" id="captureNote">The main frame becomes your confirmation preview after you capture or upload an image.</p>

        <div class="control-stack">
          <div class="control-row">
            <div class="field-group">
              <label for="cropMode">Preprocessing strategy</label>
              <select id="cropMode">
                <option value="face" selected>Face priority</option>
                <option value="tight">Tight crop fallback</option>
                <option value="portrait">Portrait crop</option>
                <option value="square">Square crop</option>
                <option value="full">Full image</option>
              </select>
            </div>
          </div>

          <div class="control-row">
            <button id="startCamera" class="primary">Start Camera</button>
            <button id="takePicture" class="secondary" disabled>Take Picture</button>
            <button id="retakePicture" class="secondary" disabled>Retake</button>
            <button id="confirmPicture" class="primary" disabled>Analyze Snapshot</button>
            <label for="uploadInput" class="upload-chip">Upload Image</label>
            <input id="uploadInput" type="file" accept="image/*" hidden />
          </div>
        </div>

        <p class="status" id="status">Ready. Start the camera or upload an image to begin.</p>
        <div class="utility-note">
          Tip: the browser will try a face-aware crop before the image reaches the server. If browser-side face detection is unavailable, the app falls back to a centered portrait crop so the model still gets a focused view.
        </div>
      </section>

      <aside class="panel">
        <div class="summary-card" id="resultCard">
          <span class="summary-label">Primary read</span>
          <div class="summary-emotion">Waiting for a confirmed image</div>
          <div class="summary-copy">Once you analyze a snapshot, the strongest emotion, its confidence, and the reasoning-oriented summary will appear here.</div>
          <div class="meta-grid">
            <div class="meta-chip"><span>Model</span><strong>ResNet18</strong></div>
            <div class="meta-chip"><span>Focus</span><strong>Face-aware crop</strong></div>
            <div class="meta-chip"><span>Output</span><strong>Top candidates</strong></div>
            <div class="meta-chip"><span>Status</span><strong>Ready</strong></div>
          </div>
        </div>

        <div class="candidate-panel" id="candidatePanel">
          <h3>Candidate emotions</h3>
          <p>The strongest competing emotions will appear here with confidence bars once the snapshot is processed.</p>
          <div class="candidate-list" id="candidateList"></div>
        </div>

        <div class="preview-grid">
          <div class="preview-card">
            <h3>Model input</h3>
            <span class="pill" id="inputTag">Awaiting analysis</span>
            <p>This is the actual processed grayscale crop sent through the model.</p>
            <div id="modelInputPlaceholder" class="preview-placeholder">
              The model input preview will appear here after the snapshot is analyzed.
            </div>
            <img id="modelInputPreview" alt="Model input preview" hidden />
          </div>
          <div class="preview-card">
            <h3>Grad-CAM overlay</h3>
            <span class="pill" id="camTag">Awaiting analysis</span>
            <p>This heatmap highlights which parts of the processed input most influenced the final prediction.</p>
            <div id="gradCamPlaceholder" class="preview-placeholder">
              The Grad-CAM explanation will appear here after the prediction finishes.
            </div>
            <img id="gradCamPreview" alt="Grad-CAM overlay" hidden />
          </div>
        </div>
      </aside>
    </div>
  </div>

  <script>
    const video = document.getElementById('video');
    const snapshot = document.getElementById('snapshot');
    const canvas = document.getElementById('canvas');
    const framePlaceholder = document.getElementById('framePlaceholder');
    const captureNote = document.getElementById('captureNote');
    const statusEl = document.getElementById('status');
    const cropModeEl = document.getElementById('cropMode');
    const startCameraBtn = document.getElementById('startCamera');
    const takePictureBtn = document.getElementById('takePicture');
    const retakePictureBtn = document.getElementById('retakePicture');
    const confirmPictureBtn = document.getElementById('confirmPicture');
    const uploadInput = document.getElementById('uploadInput');
    const resultCard = document.getElementById('resultCard');
    const candidateList = document.getElementById('candidateList');
    const modelInputPreview = document.getElementById('modelInputPreview');
    const gradCamPreview = document.getElementById('gradCamPreview');
    const modelInputPlaceholder = document.getElementById('modelInputPlaceholder');
    const gradCamPlaceholder = document.getElementById('gradCamPlaceholder');
    const inputTag = document.getElementById('inputTag');
    const camTag = document.getElementById('camTag');

    const supportsBrowserFaceDetection = 'FaceDetector' in window;
    const prettyCropNames = {
      face: 'Face priority',
      tight: 'Tight crop fallback',
      portrait: 'Portrait crop',
      square: 'Square crop',
      full: 'Full image',
    };

    let cameraStream = null;
    let capturedDataUrl = null;

    function setStatus(message, warning = false) {
      statusEl.textContent = message;
      statusEl.classList.toggle('warning', warning);
    }

    function setCaptureState({ cameraReady, hasSnapshot, predicting }) {
      takePictureBtn.disabled = !cameraReady || predicting;
      retakePictureBtn.disabled = !hasSnapshot || predicting;
      confirmPictureBtn.disabled = !hasSnapshot || predicting;
      startCameraBtn.disabled = predicting;
      cropModeEl.disabled = predicting;
      uploadInput.disabled = predicting;
    }

    function showFramePlaceholder(message) {
      framePlaceholder.innerHTML = `<div><strong>No image selected yet</strong>${message}</div>`;
      framePlaceholder.hidden = false;
      video.hidden = true;
      snapshot.hidden = true;
      snapshot.removeAttribute('src');
      captureNote.textContent = 'The main frame becomes your confirmation preview after you capture or upload an image.';
    }

    function showLivePreview() {
      framePlaceholder.hidden = true;
      snapshot.hidden = true;
      snapshot.removeAttribute('src');
      video.hidden = false;
      captureNote.textContent = 'You are looking at the live camera preview. Take a picture when the framing feels right.';
    }

    function showCapturedSnapshot(dataUrl, source = 'snapshot') {
      snapshot.src = dataUrl;
      snapshot.hidden = false;
      video.hidden = true;
      framePlaceholder.hidden = true;
      captureNote.textContent = source === 'upload'
        ? 'This uploaded image is waiting for confirmation.'
        : 'This camera snapshot is waiting for confirmation.';
    }

    function resetResultPanels() {
      resultCard.innerHTML = `
        <span class="summary-label">Primary read</span>
        <div class="summary-emotion">Waiting for a confirmed image</div>
        <div class="summary-copy">Once you analyze a snapshot, the strongest emotion, its confidence, and the reasoning-oriented summary will appear here.</div>
        <div class="meta-grid">
          <div class="meta-chip"><span>Model</span><strong>ResNet18</strong></div>
          <div class="meta-chip"><span>Focus</span><strong>Face-aware crop</strong></div>
          <div class="meta-chip"><span>Output</span><strong>Top candidates</strong></div>
          <div class="meta-chip"><span>Status</span><strong>Ready</strong></div>
        </div>
      `;
      candidateList.innerHTML = '';
      modelInputPreview.hidden = true;
      gradCamPreview.hidden = true;
      modelInputPreview.removeAttribute('src');
      gradCamPreview.removeAttribute('src');
      modelInputPlaceholder.hidden = false;
      gradCamPlaceholder.hidden = false;
      inputTag.textContent = 'Awaiting analysis';
      camTag.textContent = 'Awaiting analysis';
    }

    function renderCandidates(topPredictions) {
      candidateList.innerHTML = topPredictions.map(item => {
        const confidence = (item.confidence * 100).toFixed(1);
        return `
          <div class="candidate-row">
            <div class="candidate-head">
              <strong>${item.emotion}</strong>
              <span>${confidence}%</span>
            </div>
            <div class="candidate-bar"><div class="candidate-fill" style="width:${confidence}%"></div></div>
          </div>
        `;
      }).join('');
    }

    function renderResult(payload) {
      const browserStrategy = payload.browser_crop_strategy || 'server-side preprocessing';
      resultCard.innerHTML = `
        <span class="summary-label">Primary read</span>
        <div class="summary-emotion">${payload.predicted_emotion}</div>
        <div class="summary-confidence">Confidence ${(payload.confidence * 100).toFixed(1)}%</div>
        <div class="summary-copy">${payload.emotion_copy}</div>
        <div class="meta-grid">
          <div class="meta-chip"><span>Model</span><strong>${payload.model}</strong></div>
          <div class="meta-chip"><span>Device</span><strong>${payload.device}</strong></div>
          <div class="meta-chip"><span>Crop mode</span><strong>${prettyCropNames[payload.crop_mode] || payload.crop_mode}</strong></div>
          <div class="meta-chip"><span>Face detected</span><strong>${payload.face_detected ? 'Yes' : 'Fallback crop'}</strong></div>
          <div class="meta-chip"><span>Browser crop</span><strong>${browserStrategy}</strong></div>
          <div class="meta-chip"><span>Status</span><strong>Prediction complete</strong></div>
        </div>
      `;

      renderCandidates(payload.top_predictions);
      modelInputPlaceholder.hidden = true;
      gradCamPlaceholder.hidden = true;
      modelInputPreview.src = payload.model_input_image;
      modelInputPreview.hidden = false;
      gradCamPreview.src = payload.grad_cam_overlay;
      gradCamPreview.hidden = false;
      inputTag.textContent = 'Exact model input';
      camTag.textContent = 'Prediction explanation';
    }

    function drawCropToDataUrl(source, box) {
      const outputCanvas = document.createElement('canvas');
      outputCanvas.width = Math.max(1, Math.round(box.width));
      outputCanvas.height = Math.max(1, Math.round(box.height));
      const context = outputCanvas.getContext('2d');
      context.drawImage(
        source,
        box.x,
        box.y,
        box.width,
        box.height,
        0,
        0,
        outputCanvas.width,
        outputCanvas.height,
      );
      return outputCanvas.toDataURL('image/jpeg', 0.92);
    }

    function buildFallbackBox(width, height) {
      const cropWidth = width * 0.56;
      const cropHeight = height * 0.68;
      const x = (width - cropWidth) / 2;
      const y = height * 0.18;
      return {
        x: Math.max(0, x),
        y: Math.max(0, y),
        width: Math.min(cropWidth, width),
        height: Math.min(cropHeight, height - Math.max(0, y)),
      };
    }

    function expandFaceBox(box, width, height) {
      const padding = Math.max(box.width, box.height) * 0.28;
      const x = Math.max(0, box.x - padding);
      const y = Math.max(0, box.y - padding);
      const right = Math.min(width, box.x + box.width + padding);
      const bottom = Math.min(height, box.y + box.height + padding);
      return { x, y, width: right - x, height: bottom - y };
    }

    async function prepareDataUrlForPrediction(dataUrl) {
      const selectedMode = cropModeEl.value;
      if (selectedMode !== 'face') {
        return {
          dataUrl,
          cropMode: selectedMode,
          browserCropStrategy: `server ${selectedMode}`,
        };
      }

      const blob = await (await fetch(dataUrl)).blob();
      const bitmap = await createImageBitmap(blob);

      if (supportsBrowserFaceDetection) {
        try {
          const detector = new FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
          const faces = await detector.detect(bitmap);
          if (faces.length > 0) {
            const faceBox = expandFaceBox(faces[0].boundingBox, bitmap.width, bitmap.height);
            return {
              dataUrl: drawCropToDataUrl(bitmap, faceBox),
              cropMode: 'full',
              browserCropStrategy: 'browser face detector',
            };
          }
        } catch (error) {
          console.warn('Browser face detection failed, using fallback crop.', error);
        }
      }

      return {
        dataUrl: drawCropToDataUrl(bitmap, buildFallbackBox(bitmap.width, bitmap.height)),
        cropMode: 'full',
        browserCropStrategy: supportsBrowserFaceDetection ? 'browser fallback crop' : 'manual fallback crop',
      };
    }

    async function predictConfirmedPhoto() {
      if (!capturedDataUrl) {
        setStatus('Capture or upload an image first.', true);
        return;
      }

      setCaptureState({ cameraReady: Boolean(cameraStream), hasSnapshot: true, predicting: true });
      setStatus('Sending the confirmed image for prediction...');

      try {
        const prepared = await prepareDataUrlForPrediction(capturedDataUrl);
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: prepared.dataUrl,
            crop_mode: prepared.cropMode,
            top_k: 3,
            browser_crop_strategy: prepared.browserCropStrategy,
          }),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(errorText || 'Prediction request failed.');
        }

        const payload = await response.json();
        renderResult(payload);
        setStatus('Prediction complete. Review the candidate emotions and Grad-CAM panel on the right.');
      } catch (error) {
        setStatus(`Prediction failed: ${error.message}`, true);
      } finally {
        setCaptureState({ cameraReady: Boolean(cameraStream), hasSnapshot: true, predicting: false });
      }
    }

    function readUploadedFile(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error('Could not read the uploaded image.'));
        reader.readAsDataURL(file);
      });
    }

    startCameraBtn.addEventListener('click', async () => {
      try {
        if (cameraStream) {
          cameraStream.getTracks().forEach(track => track.stop());
        }
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = cameraStream;
        capturedDataUrl = null;
        uploadInput.value = '';
        resetResultPanels();
        showLivePreview();
        setCaptureState({ cameraReady: true, hasSnapshot: false, predicting: false });
        setStatus('Camera started. Center your face and take a picture when the framing feels good.');
      } catch (error) {
        setStatus(`Could not start camera: ${error.message}`, true);
      }
    });

    takePictureBtn.addEventListener('click', () => {
      if (!cameraStream) {
        setStatus('Start the camera first.', true);
        return;
      }

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const context = canvas.getContext('2d');
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      capturedDataUrl = canvas.toDataURL('image/jpeg', 0.92);
      showCapturedSnapshot(capturedDataUrl, 'camera');
      setCaptureState({ cameraReady: true, hasSnapshot: true, predicting: false });
      setStatus('Snapshot captured. If this is the frame you want, click Analyze Snapshot.');
    });

    retakePictureBtn.addEventListener('click', () => {
      capturedDataUrl = null;
      resetResultPanels();
      if (cameraStream) {
        showLivePreview();
        setCaptureState({ cameraReady: true, hasSnapshot: false, predicting: false });
        setStatus('Retake ready. Adjust your framing and capture another image.');
      } else {
        showFramePlaceholder('Start the camera to open the live preview, or upload an image instead.');
        setCaptureState({ cameraReady: false, hasSnapshot: false, predicting: false });
        setStatus('Ready for a new image.');
      }
      uploadInput.value = '';
    });

    confirmPictureBtn.addEventListener('click', predictConfirmedPhoto);

    uploadInput.addEventListener('change', async event => {
      const file = event.target.files && event.target.files[0];
      if (!file) return;

      try {
        capturedDataUrl = await readUploadedFile(file);
        resetResultPanels();
        showCapturedSnapshot(capturedDataUrl, 'upload');
        setCaptureState({ cameraReady: Boolean(cameraStream), hasSnapshot: true, predicting: false });
        setStatus('Upload ready. If this is the portrait you want, click Analyze Snapshot.');
      } catch (error) {
        setStatus(error.message, true);
      }
    });

    resetResultPanels();
    showFramePlaceholder('Start the camera to open the live preview, or upload an image instead. After you take or choose an image, that exact still will remain here until you confirm it.');
    setCaptureState({ cameraReady: false, hasSnapshot: false, predicting: false });
  </script>
</body>
</html>
    """


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        image = decode_data_url(payload.image)
        runtime = load_runtime(DEFAULT_MODEL_NAME, str(DEFAULT_WEIGHTS_PATH))
        return runtime.predict_with_gradcam(
            image,
            crop_mode=payload.crop_mode,
            top_k=payload.top_k,
            browser_crop_strategy=payload.browser_crop_strategy,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:  # pragma: no cover - keep browser responses readable during demo debugging.
        raise HTTPException(status_code=500, detail=f"Unexpected prediction error: {error}") from error

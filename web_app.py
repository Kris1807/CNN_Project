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
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "resnet18")
DEFAULT_WEIGHTS_PATH = Path(os.getenv("MODEL_WEIGHTS", "best_resnet18.pt"))
DEFAULT_CROP_MODE = os.getenv("CROP_MODE", "face")

app = FastAPI(title="FER Emotion Web App")


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


# Use a small NumPy-based jet-style colormap so the web deployment does not need matplotlib.
def apply_jet_colormap(cam_map: np.ndarray) -> np.ndarray:
    cam = np.clip(cam_map, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


# Blend the normalized Grad-CAM map onto the model input image so the result is easy to interpret in the UI.
def build_overlay_image(display_image: np.ndarray, cam_map: np.ndarray) -> Image.Image:
    heat = apply_jet_colormap(cam_map)
    overlay = np.clip(0.55 * display_image + 0.45 * heat, 0, 1)
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
  <title>Emotion Recognition Snapshot Demo</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f172a;
      --panel: #111827;
      --accent: #22c55e;
      --muted: #94a3b8;
      --text: #e5e7eb;
      --border: rgba(148, 163, 184, 0.22);
      --warning: #f59e0b;
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #1e293b, var(--bg) 55%);
      color: var(--text);
      min-height: 100vh;
      padding: 28px 18px;
    }
    .shell {
      max-width: 1220px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 24px;
    }
    .panel {
      background: rgba(17, 24, 39, 0.92);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 20px;
      box-shadow: 0 24px 60px rgba(0, 0, 0, 0.32);
    }
    h1, h2, h3 { margin: 0 0 10px; }
    p { color: var(--muted); line-height: 1.5; }
    .step-list {
      display: grid;
      gap: 10px;
      margin: 16px 0 0;
      padding: 0;
      list-style: none;
    }
    .step-list li {
      border: 1px solid rgba(34, 197, 94, 0.18);
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(15, 23, 42, 0.72);
    }
    .frame {
      width: 100%;
      aspect-ratio: 4 / 3;
      background: #020617;
      border-radius: 18px;
      overflow: hidden;
      display: grid;
      place-items: center;
      border: 1px solid rgba(148, 163, 184, 0.18);
      margin: 18px 0;
      position: relative;
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
      display: grid;
      place-items: center;
      text-align: center;
      padding: 24px;
      color: var(--muted);
      background:
        radial-gradient(circle at 50% 20%, rgba(34, 197, 94, 0.12), transparent 38%),
        linear-gradient(180deg, rgba(15, 23, 42, 0.78), rgba(2, 6, 23, 0.94));
      z-index: 2;
    }
    .empty-state strong {
      display: block;
      color: var(--text);
      margin-bottom: 8px;
      font-size: 1.05rem;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 12px;
      align-items: center;
    }
    button, select {
      border-radius: 12px;
      border: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.95);
      color: var(--text);
      padding: 10px 14px;
      font: inherit;
    }
    button {
      cursor: pointer;
      background: linear-gradient(135deg, #16a34a, var(--accent));
      color: #052e16;
      font-weight: 700;
      border: none;
    }
    button.secondary {
      background: transparent;
      color: var(--text);
      border: 1px solid var(--border);
      font-weight: 600;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.45;
    }
    .status {
      font-size: 0.95rem;
      color: var(--muted);
      min-height: 1.5em;
      margin-top: 12px;
    }
    .warning {
      color: var(--warning);
      font-weight: 600;
    }
    .result-card {
      margin-top: 16px;
      padding: 16px;
      border-radius: 18px;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(34, 197, 94, 0.22);
    }
    .result-card strong {
      font-size: 1.15rem;
    }
    .top-list {
      margin: 12px 0 0;
      padding-left: 22px;
    }
    .preview-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 18px;
    }
    .preview-card {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 16px;
      padding: 12px;
    }
    .preview-card img {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      border-radius: 12px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      margin-top: 8px;
      background: #020617;
    }
    .preview-placeholder {
      margin-top: 8px;
      aspect-ratio: 1 / 1;
      border-radius: 12px;
      border: 1px dashed rgba(148, 163, 184, 0.24);
      display: grid;
      place-items: center;
      text-align: center;
      padding: 18px;
      color: var(--muted);
      background: rgba(2, 6, 23, 0.55);
    }
    .capture-note {
      margin-top: 10px;
      font-size: 0.92rem;
      color: var(--muted);
    }
    .pill {
      display: inline-block;
      border-radius: 999px;
      padding: 4px 10px;
      background: rgba(34, 197, 94, 0.16);
      color: #bbf7d0;
      font-size: 0.82rem;
      margin-top: 4px;
    }
    @media (max-width: 960px) {
      .shell { grid-template-columns: 1fr; }
      .preview-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="panel">
      <h1>Emotion Recognition Snapshot Demo</h1>

      <ol class="step-list">
        <li><strong>1.</strong> Start the camera and center your face in the frame.</li>
        <li><strong>2.</strong> Take a picture when the framing looks good.</li>
        <li><strong>3.</strong> Confirm the still image. The model will predict from that exact photo.</li>
      </ol>

      <div class="frame">
        <video id="video" autoplay playsinline muted hidden></video>
        <img id="snapshot" alt="Captured snapshot" hidden />
        <canvas id="canvas"></canvas>
        <div id="framePlaceholder" class="empty-state">
          <div>
            <strong>No snapshot yet</strong>
            Start the camera to open the live preview. After you take a picture, the exact captured image will stay here until you confirm it.
          </div>
        </div>
      </div>
      <p class="capture-note" id="captureNote">The large frame becomes your confirmation preview after you take a picture.</p>

      <div class="controls">
        <button id="startCamera">Start Camera</button>
        <button id="takePicture" disabled>Take Picture</button>
        <button id="retakePicture" class="secondary" disabled>Retake</button>
        <button id="confirmPicture" disabled>Predict Confirmed Photo</button>
      </div>


      <p class="status" id="status">Ready. Start the camera to begin.</p>
    </section>

    <aside class="panel">
      <h2>Prediction Result</h2>
      <div id="result" class="result-card">
        <p>No confirmed photo has been analyzed yet.</p>
      </div>

      <div class="preview-grid">
        <div class="preview-card">
          <h3>Photo Sent To Model</h3>
          <span class="pill" id="inputTag">Waiting for prediction</span>
          <div id="modelInputPlaceholder" class="preview-placeholder">
            The processed grayscale crop will appear here after you confirm a snapshot.
          </div>
          <img id="modelInputPreview" alt="Model input preview" hidden />
        </div>
        <div class="preview-card">
          <h3>Grad-CAM Result</h3>
          <span class="pill" id="camTag">Waiting for prediction</span>
          <div id="gradCamPlaceholder" class="preview-placeholder">
            The explanation heatmap will appear here after the model finishes predicting.
          </div>
          <img id="gradCamPreview" alt="Grad-CAM overlay" hidden />
        </div>
      </div>
    </aside>
  </div>

  <script>
    const video = document.getElementById('video');
    const snapshot = document.getElementById('snapshot');
    const canvas = document.getElementById('canvas');
    const framePlaceholder = document.getElementById('framePlaceholder');
    const captureNote = document.getElementById('captureNote');
    const statusEl = document.getElementById('status');
    const resultEl = document.getElementById('result');
    const fixedCropMode = 'face';
    const startCameraBtn = document.getElementById('startCamera');
    const takePictureBtn = document.getElementById('takePicture');
    const retakePictureBtn = document.getElementById('retakePicture');
    const confirmPictureBtn = document.getElementById('confirmPicture');
    const modelInputPreview = document.getElementById('modelInputPreview');
    const gradCamPreview = document.getElementById('gradCamPreview');
    const modelInputPlaceholder = document.getElementById('modelInputPlaceholder');
    const gradCamPlaceholder = document.getElementById('gradCamPlaceholder');
    const inputTag = document.getElementById('inputTag');
    const camTag = document.getElementById('camTag');

    let cameraStream = null;
    let capturedDataUrl = null;
    const supportsBrowserFaceDetection = 'FaceDetector' in window;

    function setStatus(message, warning = false) {
      statusEl.textContent = message;
      statusEl.classList.toggle('warning', warning);
    }

    function setCaptureState({ cameraReady, hasSnapshot, predicting }) {
      takePictureBtn.disabled = !cameraReady || predicting;
      retakePictureBtn.disabled = !hasSnapshot || predicting;
      confirmPictureBtn.disabled = !hasSnapshot || predicting;
      startCameraBtn.disabled = predicting;
    }

    function showFramePlaceholder(message) {
      framePlaceholder.innerHTML = `<div><strong>No snapshot yet</strong>${message}</div>`;
      framePlaceholder.hidden = false;
      video.hidden = true;
      snapshot.hidden = true;
      snapshot.removeAttribute('src');
      captureNote.textContent = 'The large frame becomes your confirmation preview after you take a picture.';
    }

    function showLivePreview() {
      framePlaceholder.hidden = true;
      snapshot.hidden = true;
      snapshot.removeAttribute('src');
      video.hidden = false;
      captureNote.textContent = 'You are looking at the live camera preview. Take a picture when the framing looks right.';
    }

    function showCapturedSnapshot(dataUrl) {
      snapshot.src = dataUrl;
      snapshot.hidden = false;
      video.hidden = true;
      framePlaceholder.hidden = true;
      captureNote.textContent = 'This is the exact photo waiting for your confirmation.';
    }

    function resetResultPanels() {
      resultEl.innerHTML = '<p>No confirmed photo has been analyzed yet.</p>';
      modelInputPreview.hidden = true;
      gradCamPreview.hidden = true;
      modelInputPreview.removeAttribute('src');
      gradCamPreview.removeAttribute('src');
      modelInputPlaceholder.hidden = false;
      gradCamPlaceholder.hidden = false;
      inputTag.textContent = 'Waiting for prediction';
      camTag.textContent = 'Waiting for prediction';
    }

    function renderResult(payload) {
      const items = payload.top_predictions
        .map(item => `<li><strong>${item.emotion}</strong>: ${(item.confidence * 100).toFixed(2)}%</li>`)
        .join('');

      const browserCropLine = payload.browser_crop_strategy
        ? `<p>Browser crop: ${payload.browser_crop_strategy}</p>`
        : '';

      resultEl.innerHTML = `
        <strong>${payload.predicted_emotion}</strong>
        <p>Confidence: ${(payload.confidence * 100).toFixed(2)}%</p>
        <p>Server face detected: ${payload.face_detected ? 'yes' : 'no, used fallback crop'}</p>
        ${browserCropLine}
        <ol class="top-list">${items}</ol>
      `;

      modelInputPlaceholder.hidden = true;
      gradCamPlaceholder.hidden = true;
      modelInputPreview.src = payload.model_input_image;
      modelInputPreview.hidden = false;
      gradCamPreview.src = payload.grad_cam_overlay;
      gradCamPreview.hidden = false;
      inputTag.textContent = 'Exact image after preprocessing';
      camTag.textContent = 'Predicted class explanation';
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
        setStatus('Take a picture first.');
        return;
      }

      setCaptureState({ cameraReady: true, hasSnapshot: true, predicting: true });
      setStatus('Sending confirmed photo for prediction...');
      resultEl.innerHTML = '<p>Processing confirmed photo...</p>';

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
        setStatus('Prediction complete.');
      } catch (error) {
        setStatus(`Prediction failed: ${error.message}`, true);
      } finally {
        setCaptureState({ cameraReady: true, hasSnapshot: true, predicting: false });
      }
    }

    startCameraBtn.addEventListener('click', async () => {
      try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = cameraStream;
        capturedDataUrl = null;
        resetResultPanels();
        showLivePreview();
        setCaptureState({ cameraReady: true, hasSnapshot: false, predicting: false });
        setStatus('Camera started. Center your face, then take a picture.');
      } catch (error) {
        setStatus(`Could not start camera: ${error.message}`, true);
      }
    });

    takePictureBtn.addEventListener('click', () => {
      if (!cameraStream) {
        setStatus('Start the camera first.');
        return;
      }

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const context = canvas.getContext('2d');
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      capturedDataUrl = canvas.toDataURL('image/jpeg', 0.92);
      showCapturedSnapshot(capturedDataUrl);
      setCaptureState({ cameraReady: true, hasSnapshot: true, predicting: false });
      setStatus('Snapshot captured. If this is the photo you want to analyze, click Predict Confirmed Photo.');
    });

    retakePictureBtn.addEventListener('click', () => {
      if (!cameraStream) {
        setStatus('Start the camera first.');
        return;
      }

      capturedDataUrl = null;
      resetResultPanels();
      showLivePreview();
      setCaptureState({ cameraReady: true, hasSnapshot: false, predicting: false });
      setStatus('Retake ready. Adjust your framing and take another picture.');
    });

    confirmPictureBtn.addEventListener('click', predictConfirmedPhoto);

    resetResultPanels();
    showFramePlaceholder('Start the camera to open the live preview. After you take a picture, the exact captured image will stay here until you confirm it.');
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

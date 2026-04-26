import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models as tv_models

from image_preprocessing import build_inference_transform
from models import SimpleCNN


# This class order matches the folder-based FER training setup used in the project.
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class LiveEmotionCamera:
    """Run live webcam inference with optional Grad-CAM overlays."""

    def __init__(
        self,
        weights_path,
        model_name="resnet18",
        camera_index=0,
        device=None,
        top_k=3,
        history=5,
        frame_skip=1,
        mirror=True,
        window_name="Live Emotion Recognition",
        use_grad_cam=False,
        grad_cam_alpha=0.45,
    ):
        self.weights_path = Path(weights_path)
        self.model_name = model_name
        self.camera_index = camera_index
        self.device = device or self._select_device()
        self.top_k = top_k
        self.history = max(1, history)
        self.frame_skip = max(1, frame_skip)
        self.mirror = mirror
        self.window_name = window_name
        # Grad-CAM is optional because it adds a backward pass and makes the live slower.
        self.use_grad_cam = use_grad_cam
        self.grad_cam_alpha = float(np.clip(grad_cam_alpha, 0.0, 1.0))

        self.transform = build_inference_transform(model_name)
        self.model, self.target_layer = self._load_model()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.probability_history = deque(maxlen=self.history)
        self.activations = None
        self.gradients = None

        if self.use_grad_cam:
            self._register_hooks()

    def _select_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # Load the same best checkpoint used by the other code sections.
    def _load_model(self):
        if self.model_name == "cnn":
            model = SimpleCNN()
            target_layer = model.features[-2]
        else:
            model = tv_models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
            target_layer = model.layer4[-1].conv2

        state = torch.load(self.weights_path, map_location=self.device)
        model.load_state_dict(state)
        model = model.to(self.device)
        model.eval()
        return model, target_layer

    # Hooks capture the feature maps and gradients needed to build a Grad-CAM heatmap.
    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    # Detect the most visible face and return a padded bounding box in image coordinates.
    # This feature helps the user, model and the Grad-CAM focus on the face instead of the whole frame.
    def detect_face_bbox(self, frame_rgb):
        grayscale = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        if len(faces) == 0:
            return None

        x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
        padding = int(max(width, height) * 0.25)
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(frame_rgb.shape[1], x + width + padding)
        bottom = min(frame_rgb.shape[0], y + height + padding)
        return left, top, right, bottom

    # Convert the cropped face into the same tensor format used during model training and testing.
    def prepare_face_tensor(self, face_rgb):
        face_image = Image.fromarray(face_rgb)
        tensor = self.transform(face_image).unsqueeze(0)
        return tensor.to(self.device)

    # Build a normalized Grad-CAM map for the chosen class on the current frame.
    # Honestly, I didn't think I could make it work- but some research reveled a GEM.
    def _generate_cam_map(self, face_tensor, class_index):
        score = self.current_logits[0, class_index]
        self.model.zero_grad(set_to_none=True)
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=face_tensor.shape[-2:], mode="bilinear", align_corners=False)

        cam_map = cam[0, 0]
        cam_map = cam_map - cam_map.min()
        cam_map = cam_map / (cam_map.max() + 1e-8)
        return cam_map.detach().cpu().numpy()

    # Average probabilities over recent frames so the on-screen label is less jumpy.
    # When Grad-CAM is enabled, this method also computes a heatmap for the predicted class.
    def predict_face(self, face_rgb):
        face_tensor = self.prepare_face_tensor(face_rgb)

        # Grad-CAM needs gradients, so we keep autograd active only in that mode.
        if self.use_grad_cam:
            self.current_logits = self.model(face_tensor)
            probabilities_np = torch.softmax(self.current_logits, dim=1)[0].detach().cpu().numpy()
        else:
            with torch.no_grad():
                self.current_logits = self.model(face_tensor)
                probabilities_np = torch.softmax(self.current_logits, dim=1)[0].detach().cpu().numpy()

        self.probability_history.append(probabilities_np)
        smoothed = np.mean(np.stack(self.probability_history), axis=0)

        class_index = int(np.argmax(smoothed))
        confidence = float(smoothed[class_index])
        top_count = min(self.top_k, len(CLASS_NAMES))
        top_indices = np.argsort(smoothed)[-top_count:][::-1]

        cam_map = None
        if self.use_grad_cam:
            cam_map = self._generate_cam_map(face_tensor, class_index)

        return {
            "predicted_emotion": CLASS_NAMES[class_index],
            "confidence": confidence,
            "top_predictions": [
                {
                    "emotion": CLASS_NAMES[int(index)],
                    "confidence": float(smoothed[int(index)]),
                }
                for index in top_indices
            ],
            "cam_map": cam_map,
        }

    # Overlay the heatmap only on the detected face region instead of the whole frame.
    # Show the user in an easier way.
    def _apply_grad_cam_overlay(self, frame_bgr, bbox, cam_map):
        if cam_map is None:
            return frame_bgr

        left, top, right, bottom = bbox
        region = frame_bgr[top:bottom, left:right]
        if region.size == 0:
            return frame_bgr

        heatmap = cv2.resize((cam_map * 255).astype(np.uint8), (region.shape[1], region.shape[0]))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(region, 1.0 - self.grad_cam_alpha, heatmap, self.grad_cam_alpha, 0)
        frame_bgr[top:bottom, left:right] = blended
        return frame_bgr

    # Draw the bounding box, predicted label and confidence of the emotions next to the frame.
    def _draw_prediction(self, frame_bgr, bbox, prediction):
        left, top, right, bottom = bbox
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

        label = f"{prediction['predicted_emotion']} ({prediction['confidence']:.2f})"
        text_y = max(30, top - 10)
        cv2.putText(
            frame_bgr,
            label,
            (left, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        detail_y = bottom + 25
        for item in prediction["top_predictions"]:
            detail = f"{item['emotion']}: {item['confidence']:.2f}"
            cv2.putText(
                frame_bgr,
                detail,
                (left, detail_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            detail_y += 22


    # Also learned that this is how a live "Main" for webcam works.
    # Main webcam loop: capture frame, detect face, predict emotion, optionally overlay Grad-CAM, and display result.
    def run(self):
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open webcam index {self.camera_index}. Check camera permissions and index."
            )

        print(f"Using device: {self.device}")
        print(f"Model: {self.model_name}")
        print(f"Grad-CAM: {'on' if self.use_grad_cam else 'off'}")
        print("Press 'q' to quit the webcam window.")
        if self.use_grad_cam:
            print("Live Grad-CAM is heavier than plain prediction, so lower FPS is expected.")

        frame_counter = 0
        last_prediction = None

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    raise RuntimeError("Failed to read a frame from the webcam.")

                if self.mirror:
                    frame_bgr = cv2.flip(frame_bgr, 1)

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                bbox = self.detect_face_bbox(frame_rgb)

                if bbox is None:
                    self.probability_history.clear()
                    last_prediction = None
                    cv2.putText(
                        frame_bgr,
                        "No face detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    left, top, right, bottom = bbox
                    face_rgb = frame_rgb[top:bottom, left:right]

                    if face_rgb.size > 0 and (frame_counter % self.frame_skip == 0 or last_prediction is None):
                        last_prediction = self.predict_face(face_rgb)

                    if last_prediction is not None:
                        if self.use_grad_cam:
                            frame_bgr = self._apply_grad_cam_overlay(frame_bgr, bbox, last_prediction["cam_map"])
                        self._draw_prediction(frame_bgr, bbox, last_prediction)

                cv2.putText(
                    frame_bgr,
                    "Press q to quit",
                    (20, frame_bgr.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(self.window_name, frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                frame_counter += 1
        finally:
            capture.release()
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the trained facial emotion model on a live webcam feed."
    )
    parser.add_argument("--weights", required=True, help="Path to the trained .pt checkpoint")
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="resnet18")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index to open")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many of the strongest emotion scores to display",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="Number of recent frames to average for smoother predictions",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Only run model inference every N frames to reduce CPU load",
    )
    parser.add_argument(
        "--grad-cam",
        action="store_true",
        help="Overlay Grad-CAM on the detected face while the webcam is running",
    )
    parser.add_argument(
        "--grad-cam-alpha",
        type=float,
        default=0.45,
        help="Blend strength for the Grad-CAM heatmap overlay (0.0 to 1.0)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable left-right mirroring of the camera feed",
    )
    parser.add_argument(
        "--window-name",
        default="Live Emotion Recognition",
        help="Window title for the webcam preview",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = LiveEmotionCamera(
        weights_path=args.weights,
        model_name=args.model,
        camera_index=args.camera_index,
        top_k=args.top_k,
        history=args.history,
        frame_skip=args.frame_skip,
        mirror=not args.no_mirror,
        window_name=args.window_name,
        use_grad_cam=args.grad_cam,
        grad_cam_alpha=args.grad_cam_alpha,
    )
    app.run()


if __name__ == "__main__":
    main()

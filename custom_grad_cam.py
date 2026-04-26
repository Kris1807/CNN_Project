import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models as tv_models
from torchvision import transforms
import cv2

from models import SimpleCNN


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_FACE_AREA_RATIO = 0.08


# Try to isolate the face before running Grad-CAM so the explanation focuses on facial cues, not background.
def detect_face_crop(image):
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    grayscale = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        return None

    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    face_area_ratio = (width * height) / (image.size[0] * image.size[1])
    if face_area_ratio < MIN_FACE_AREA_RATIO:
        return None

    padding = int(max(width, height) * 0.25)
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image.size[0], x + width + padding)
    bottom = min(image.size[1], y + height + padding)
    return image.crop((left, top, right, bottom))


# Multiple crop strategies make it possible to compare whole-image inference against face-focused inference.
def apply_crop_mode(image, crop_mode):
    width, height = image.size

    if crop_mode == "face":
        detected = detect_face_crop(image)
        if detected is not None:
            return detected
        # If face detection fails, fall back to a tighter manual crop instead of aborting.
        return apply_crop_mode(image, "tight")

    if crop_mode == "full":
        return image

    if crop_mode == "square":
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        return image.crop((left, top, left + side, top + side))

    if crop_mode == "portrait":
        crop_width = int(width * 0.82)
        crop_height = int(height * 0.72)
        left = max(0, (width - crop_width) // 2)
        top = max(0, int(height * 0.10))
        right = min(width, left + crop_width)
        bottom = min(height, top + crop_height)
        return image.crop((left, top, right, bottom))

    if crop_mode == "tight":
        crop_width = int(width * 0.68)
        crop_height = int(height * 0.58)
        left = max(0, (width - crop_width) // 2)
        top = max(0, int(height * 0.12))
        right = min(width, left + crop_width)
        bottom = min(height, top + crop_height)
        return image.crop((left, top, right, bottom))

    raise ValueError(f"Unsupported crop mode: {crop_mode}")


# This class mirrors the dataset Grad-CAM script, but it works on arbitrary personal images.
class CustomImageGradCAM:
    def __init__(self, weights_path, model_name="resnet18", device=None, crop_mode="face"):
        self.weights_path = Path(weights_path)
        self.model_name = model_name
        self.device = device or self._select_device()
        self.crop_mode = crop_mode
        self.model, self.transform, self.target_layer = self._load_model_and_transform()
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _select_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # Load the same trained checkpoint used elsewhere in the project and pair it with the correct preprocessing.
    def _load_model_and_transform(self):
        if self.model_name == "cnn":
            model = SimpleCNN()
            transform = transforms.Compose(
                [
                    transforms.Resize((48, 48)),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
            target_layer = model.features[-2]
        else:
            model = tv_models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            target_layer = model.layer4[-1].conv2

        state = torch.load(self.weights_path, map_location=self.device)
        model.load_state_dict(state)
        model = model.to(self.device)
        model.eval()
        return model, transform, target_layer

    # Hooks capture the intermediate feature maps and gradients required by Grad-CAM.
    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def _prepare_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = apply_crop_mode(image, self.crop_mode)
        tensor = self.transform(image).unsqueeze(0)
        return image, tensor.to(self.device)

    # Undo normalization so the saved visualization is easy to read and view vy the user.
    def _denormalize(self, tensor):
        x = tensor.detach().cpu().clone()
        if x.shape[0] == 1:
            x = x * 0.5 + 0.5
            x = x.clamp(0, 1).squeeze(0).numpy()
            return np.stack([x, x, x], axis=-1)

        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        x = x * std + mean
        return x.clamp(0, 1).permute(1, 2, 0).numpy()

    # Generate a prediction and its Grad-CAM heatmap for one custom image.
    def generate_for_image(self, image_path):
        _, image_tensor = self._prepare_image(image_path)

        logits = self.model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, class_index = probabilities.max(dim=0)

        score = logits[0, class_index]
        self.model.zero_grad(set_to_none=True)
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)

        cam_map = cam[0, 0]
        cam_map = cam_map - cam_map.min()
        cam_map = cam_map / (cam_map.max() + 1e-8)
        cam_map = cam_map.detach().cpu().numpy()

        display_image = self._denormalize(image_tensor[0])
        return {
            "image_path": str(image_path),
            "predicted_emotion": CLASS_NAMES[int(class_index.item())],
            "confidence": float(confidence.item()),
            "display_image": display_image,
            "cam_map": cam_map,
        }

    # Save the explanation in the same side-by-side format used elsewhere in the project.
    def save_visualization(self, result, out_path):
        heat = plt.cm.jet(result["cam_map"])[..., :3]
        overlay = np.clip(0.55 * result["display_image"] + 0.45 * heat, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        axes[0].imshow(result["display_image"])
        axes[0].set_title("Original")
        axes[1].imshow(result["cam_map"], cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[2].imshow(overlay)
        axes[2].set_title(
            f"{result['predicted_emotion']}\nconf:{result['confidence']:.4f}"
        )
        for ax in axes:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)


# Accept either a folder of images or an explicit list of files.
def collect_image_paths(input_dir=None, image_paths=None):
    collected_paths = []

    if input_dir:
        base_dir = Path(input_dir)
        if not base_dir.is_dir():
            raise FileNotFoundError(f"Could not find input directory: {base_dir}")
        collected_paths.extend(
            sorted(path for path in base_dir.iterdir() if path.suffix.lower() in VALID_EXTENSIONS)
        )

    if image_paths:
        collected_paths.extend(Path(path) for path in image_paths)

    unique_paths = []
    seen = set()
    for path in collected_paths:
        resolved = str(Path(path))
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(Path(path))

    return unique_paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM overlays for custom images using a trained emotion model."
    )
    parser.add_argument("--weights", required=True, help="Path to the trained .pt checkpoint")
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="resnet18")
    parser.add_argument("--input-dir", help="Folder containing custom images")
    parser.add_argument("--images", nargs="*", help="Optional list of image paths")
    parser.add_argument("--out-dir", default="custom_gradcam_outputs")
    parser.add_argument("--min-images", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--crop-mode",
        choices=["face", "full", "square", "portrait", "tight"],
        default="face",
        help="How aggressively to crop the image before Grad-CAM",
    )
    args = parser.parse_args()

    image_paths = collect_image_paths(input_dir=args.input_dir, image_paths=args.images)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    if len(image_paths) < args.min_images:
        raise ValueError(
            f"Found {len(image_paths)} image(s). Please provide at least {args.min_images} images."
        )

    missing_paths = [str(path) for path in image_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "These image paths were not found:\n" + "\n".join(missing_paths)
        )

    os.makedirs(args.out_dir, exist_ok=True)

    generator = CustomImageGradCAM(
        args.weights,
        model_name=args.model,
        crop_mode=args.crop_mode,
    )
    print(f"Using device: {generator.device}")
    print(f"Crop mode: {generator.crop_mode}")
    print(f"Generating Grad-CAM for {len(image_paths)} image(s)...")

    for index, image_path in enumerate(image_paths, start=1):
        result = generator.generate_for_image(image_path)
        image_name = Path(image_path).stem
        out_path = Path(args.out_dir) / (
            f"{index:02d}_{image_name}_{result['predicted_emotion']}.png"
        )
        generator.save_visualization(result, out_path)
        print(
            f"{result['image_path']} -> {result['predicted_emotion']} "
            f"(confidence: {result['confidence']:.4f}) | saved: {out_path}"
        )


if __name__ == "__main__":
    main()

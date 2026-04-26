import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import models as tv_models
from torchvision import transforms
import cv2

from models import SimpleCNN


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_FACE_AREA_RATIO = 0.08


# Detect the face first so custom selfies look closer to the FER-style images used during training.
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


# Different crop modes make it easier to test how framing influences emotion predictions.
def apply_crop_mode(image, crop_mode):
    width, height = image.size

    if crop_mode == "face":
        detected = detect_face_crop(image)
        if detected is not None:
            return detected
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


# This class loads the trained model and applies it to user-supplied images outside the FER ready dataset.
class EmotionImageClassifier:
    def __init__(self, weights_path, model_name="resnet18", device=None, crop_mode="face"):
        self.weights_path = Path(weights_path)
        self.model_name = model_name
        self.device = device or self._select_device()
        self.crop_mode = crop_mode
        self.model, self.transform = self._load_model_and_transform()

    def _select_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # Load the architecture and then inject the saved checkpoint weights from training.
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

        state = torch.load(self.weights_path, map_location=self.device)
        model.load_state_dict(state)
        model = model.to(self.device)
        model.eval()
        return model, transform

    # Custom images are cropped and transformed to match the format expected by the trained model.
    def _prepare_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = apply_crop_mode(image, self.crop_mode)
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    # Return both the top prediction and the top-k candidates so uncertainty is visible.
    def predict_image(self, image_path, top_k=3):
        image_tensor = self._prepare_image(image_path)
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, class_index = probabilities.max(dim=0)
            k = min(top_k, len(CLASS_NAMES))
            top_values, top_indices = probabilities.topk(k)

        return {
            "image_path": str(image_path),
            "predicted_emotion": CLASS_NAMES[int(class_index.item())],
            "confidence": float(confidence.item()),
            "top_predictions": [
                {
                    "emotion": CLASS_NAMES[int(index.item())],
                    "confidence": float(value.item()),
                }
                for value, index in zip(top_values, top_indices)
            ],
        }

    def predict_images(self, image_paths, top_k=3):
        results = []
        for image_path in image_paths:
            results.append(self.predict_image(image_path, top_k=top_k))
        return results


# Accept either an image folder or a direct list of files.
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
        description="Classify custom facial images using a trained emotion model."
    )
    parser.add_argument("--weights", required=True, help="Path to the trained .pt checkpoint")
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="resnet18")
    parser.add_argument("--input-dir", help="Folder containing custom images to classify")
    parser.add_argument(
        "--images",
        nargs="*",
        help="Optional list of individual image paths to classify",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=5,
        help="Minimum number of images required before running predictions",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on how many images to classify",
    )
    parser.add_argument(
        "--crop-mode",
        choices=["face", "full", "square", "portrait", "tight"],
        default="face",
        help="How aggressively to crop the image before prediction",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many highest-probability emotions to print for each image",
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

    classifier = EmotionImageClassifier(
        args.weights,
        model_name=args.model,
        crop_mode=args.crop_mode,
    )
    results = classifier.predict_images(image_paths, top_k=args.top_k)

    print(f"Using device: {classifier.device}")
    print(f"Crop mode: {classifier.crop_mode}")
    print(f"Classified {len(results)} images:\n")
    for result in results:
        top_summary = ", ".join(
            f"{item['emotion']}:{item['confidence']:.4f}"
            for item in result["top_predictions"]
        )
        print(f"{result['image_path']}")
        print(
            f"  predicted: {result['predicted_emotion']} "
            f"(confidence: {result['confidence']:.4f})"
        )
        print(f"  top {len(result['top_predictions'])}: {top_summary}")


if __name__ == "__main__":
    main()

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_FACE_AREA_RATIO = 0.08


# Detect the largest face in the image and add a little padding so facial context is preserved.
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


# These crop modes let the program test how strongly framing affects custom-image performance.
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


# This transform is used for inference, so it includes tensor conversion and normalization.
def build_inference_transform(model_name):
    if model_name == "cnn":
        return transforms.Compose(
            [
                transforms.Resize((48, 48)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


# This export transform prepares images for inspection only, so it leaves them as PIL images without the normalization.
def build_export_transform(model_name):
    if model_name == "cnn":
        return transforms.Compose(
            [
                transforms.Resize((48, 48)),
                transforms.Grayscale(num_output_channels=1),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
        ]
    )


# Return both the original image and the cropped image so callers can inspect the preprocessing choice.
def load_and_crop_image(image_path, crop_mode):
    image = Image.open(image_path).convert("RGB")
    return image, apply_crop_mode(image, crop_mode)


# Helper that supports both folder-based input and direct file lists,
# and it also is removing duplicates.
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

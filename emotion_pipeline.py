import os

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

from fer_dataset import EMOTIONS, FER2013Dataset
from models import SimpleCNN, build_resnet18


# Centralize model-specific settings so every script uses the same image size and normalization.
MODEL_SPECS = {
    "cnn": {
        "image_size": 48,
        "channels": 1,
        "normalize_mean": (0.5,),
        "normalize_std": (0.5,),
    },
    "resnet18": {
        "image_size": 224,
        "channels": 3,
        "normalize_mean": (0.5, 0.5, 0.5),
        "normalize_std": (0.5, 0.5, 0.5),
    },
}

# I thought about running it on a computer at the AI lab, so I prepared it to use GPU if available,
# but i never actually tried it.
# Picks the best available accelerator automatically, while still working on CPU-only machines.
def resolve_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# Keeps model construction in one place so train/evaluate/Grad-CAM all use the same architecture.
def build_model(model_name):
    if model_name == "cnn":
        return SimpleCNN()
    if model_name == "resnet18":
        return build_resnet18()
    raise ValueError(f"Unsupported model name: {model_name}")


# Training and evaluation use the same base preprocessing, with augmentation only enabled for training.
def build_transforms(model_name, training):
    spec = MODEL_SPECS[model_name]
    steps = [transforms.Resize((spec["image_size"], spec["image_size"]))]

    if spec["channels"] == 1:
        steps.append(transforms.Grayscale(num_output_channels=1))
    else:
        steps.append(transforms.Grayscale(num_output_channels=3))

    if training:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        )

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(spec["normalize_mean"], spec["normalize_std"]),
        ]
    )
    return transforms.Compose(steps)


# For folder-based datasets, create a reproducible random split between training and validation images.
def split_folder_dataset(train_dir, train_transform, eval_transform, val_ratio=0.1, seed=42):
    base_dataset = datasets.ImageFolder(train_dir)
    sample_count = len(base_dataset)
    if sample_count < 2:
        raise ValueError(f"Not enough images in {train_dir} to create a validation split")

    validation_size = max(1, int(sample_count * val_ratio))
    training_size = sample_count - validation_size

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(sample_count, generator=generator).tolist()
    train_indices = permutation[:training_size]
    validation_indices = permutation[training_size:]

    training_dataset = Subset(
        datasets.ImageFolder(train_dir, transform=train_transform),
        train_indices,
    )
    validation_dataset = Subset(
        datasets.ImageFolder(train_dir, transform=eval_transform),
        validation_indices,
    )
    class_names = datasets.ImageFolder(train_dir).classes
    return training_dataset, validation_dataset, class_names


# Support both original FER CSV input and extracted train/test folder input.
def load_training_datasets(csv_path, data_dir, model_name, val_ratio=0.1, seed=42):
    train_transform = build_transforms(model_name, training=True)
    eval_transform = build_transforms(model_name, training=False)

    if csv_path:
        return {
            "mode_label": "CSV",
            "class_names": EMOTIONS,
            "train_dataset": FER2013Dataset(csv_path, "train", transform=train_transform),
            "validation_dataset": FER2013Dataset(csv_path, "val", transform=eval_transform),
        }

    train_dir = os.path.join(data_dir, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Could not find train directory: {train_dir}. "
            "Either provide --csv or ensure --data-dir has train/ and Evaluate/."
        )

    train_dataset, validation_dataset, class_names = split_folder_dataset(
        train_dir,
        train_transform,
        eval_transform,
        val_ratio=val_ratio,
        seed=seed,
    )
    return {
        "mode_label": f"folders ({data_dir})",
        "class_names": class_names,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
    }


# Test loading mirrors training loading so evaluation always uses the same class ordering and preprocessing.
def load_test_dataset(csv_path, data_dir, model_name):
    eval_transform = build_transforms(model_name, training=False)

    if csv_path:
        return {
            "mode_label": "CSV",
            "class_names": EMOTIONS,
            "dataset": FER2013Dataset(csv_path, "test", transform=eval_transform),
        }

    test_dir = os.path.join(data_dir, "Evaluate")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Could not find test directory: {test_dir}. "
            "Either provide --csv or ensure --data-dir has train/ and Evaluate/."
        )

    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    return {
        "mode_label": f"folders ({data_dir})",
        "class_names": test_dataset.classes,
        "dataset": test_dataset,
    }

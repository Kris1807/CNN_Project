import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from emotion_pipeline import build_model, load_test_dataset, resolve_device


# Evaluation uses the same command-line style as training for consistency.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to fer2013.csv (optional)")
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Root directory containing train/ and Evaluate/ folders (used when --csv is omitted)",
    )
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="resnet18")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


# Test data is never shuffled so the prediction order stays deterministic.
def build_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# This pass only performs inference; gradients are disabled to save memory and time.
def collect_predictions(model, loader, device):
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            predicted_batch = model(images).argmax(dim=1).cpu()
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predicted_batch.tolist())

    return true_labels, predicted_labels


# Load the best saved checkpoint and report the final metrics on the held-out test set.
def evaluate_model(args):
    device = resolve_device()
    dataset_bundle = load_test_dataset(args.csv, args.data_dir, args.model)

    model = build_model(args.model)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Data mode: {dataset_bundle['mode_label']}")
    print(f"Classes: {dataset_bundle['class_names']}")

    loader = build_loader(dataset_bundle["dataset"], args.batch_size)
    true_labels, predicted_labels = collect_predictions(model, loader, device)

    print("Confusion Matrix:")
    # Its so nice that this is a method that creates that confusion matrix here and not something I had to implement myself.
    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=dataset_bundle["class_names"],
            digits=4,
        )
    )


def main():
    args = parse_args()
    evaluate_model(args)


if __name__ == "__main__":
    main()

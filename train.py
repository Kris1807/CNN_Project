import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from emotion_pipeline import build_model, load_training_datasets, resolve_device


# Keep command-line parsing in one place so the training entry point stays easy to read.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to fer2013.csv (optional)")
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Root directory containing train/ and Evaluate/ folders (used when --csv is omitted)",
    )
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="best_model.pt")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# DataLoaders handle batching and shuffling; training shuffles, validation does not.
def build_loaders(train_dataset, validation_dataset, batch_size):
    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        "validation": DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        ),
    }


# The same loop is reused for both training and validation.
# Passing an optimizer switches the function into training mode.
def run_pass(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(mode=training)

    sample_count = 0
    correct_count = 0
    summed_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        sample_count += batch_size
        summed_loss += loss.item() * batch_size
        correct_count += (logits.argmax(dim=1) == labels).sum().item()

    return {
        "loss": summed_loss / sample_count,
        "accuracy": correct_count / sample_count,
    }


# Print the resolved dataset mode so it is obvious whether training uses CSV or folders.
def print_data_summary(dataset_bundle):
    print(f"Data mode: {dataset_bundle['mode_label']}")
    print(f"Classes: {dataset_bundle['class_names']}")


# Save only when validation accuracy improves, so the checkpoint tracks the best model seen so far.
def maybe_save_checkpoint(model, checkpoint_path, validation_accuracy, best_accuracy):
    if validation_accuracy <= best_accuracy:
        return best_accuracy

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved best checkpoint to {checkpoint_path}")
    return validation_accuracy


# This is the main training workflow: load data, build the model, then iterate through epochs.
def train_model(args):
    device = resolve_device()
    dataset_bundle = load_training_datasets(
        csv_path=args.csv,
        data_dir=args.data_dir,
        model_name=args.model,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print_data_summary(dataset_bundle)

    model = build_model(args.model).to(device)
    loaders = build_loaders(
        dataset_bundle["train_dataset"],
        dataset_bundle["validation_dataset"],
        args.batch_size,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_validation_accuracy = 0.0
    for epoch_index in range(1, args.epochs + 1):
        train_metrics = run_pass(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
        )
        validation_metrics = run_pass(
            model,
            loaders["validation"],
            criterion,
            device,
        )
        print(
            f"Epoch {epoch_index:02d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
            f"val loss {validation_metrics['loss']:.4f} acc {validation_metrics['accuracy']:.4f}"
        )
        best_validation_accuracy = maybe_save_checkpoint(
            model,
            args.out,
            validation_metrics["accuracy"],
            best_validation_accuracy,
        )


def main():
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()

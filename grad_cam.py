import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fer_dataset import FER2013Dataset, EMOTIONS
from models import SimpleCNN, build_resnet18


# Convert a normalized piece back into a viewable image for plotting.
def denorm_image(tensor):
    x = tensor.detach().cpu().clone()
    if x.shape[0] == 1:
        x = x * 0.5 + 0.5
        x = x.clamp(0, 1).squeeze(0).numpy()
        return np.stack([x, x, x], axis=-1)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    x = x * std + mean
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    return x


# Grad-CAM stores activations and gradients from a chosen convolution layer, then turns them into a heatmap.
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)

        score = logits[torch.arange(logits.size(0), device=logits.device), class_idx].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        # gradient averages act as importance weights for the saved activations.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out = []
        for i in range(cam.size(0)):
            c = cam[i, 0]
            c = c - c.min()
            c = c / (c.max() + 1e-8)
            out.append(c.detach().cpu().numpy())
        return logits.detach(), np.array(out)


# Match the preprocessing and target layer to the selected architecture before generating explanations.
def build_model_and_transform(model_name, weights_path, device):
    if model_name == "cnn":
        model = SimpleCNN()
        tfm = transforms.Compose(
            [
                transforms.Resize((48, 48)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        target_layer = model.features[-2]
    else:
        model = build_resnet18()
        tfm = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        target_layer = model.layer4[-1].conv2

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model, tfm, target_layer


# Save a side-by-side figure so the original image, heatmap, and overlay can be inspected together.
def save_gradcam_figure(img_np, cam_np, true_label, pred_label, out_path):
    heat = plt.cm.jet(cam_np)[..., :3]
    overlay = np.clip(0.55 * img_np + 0.45 * heat, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[1].imshow(cam_np, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\nT:{true_label} P:{pred_label}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None, help="Path to fer2013.csv (optional)")
    p.add_argument(
        "--data-dir",
        default=".",
        help="Root directory containing train/ and Evaluate/ folders (used when --csv is omitted)",
    )
    p.add_argument("--model", choices=["cnn", "resnet18"], default="resnet18")
    p.add_argument("--weights", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--num-images", type=int, default=12)
    p.add_argument("--out-dir", default="gradcam_outputs")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tfm, target_layer = build_model_and_transform(args.model, args.weights, device)
    cam_engine = GradCAM(model, target_layer)

    if args.csv:
        ds = FER2013Dataset(args.csv, split=args.split, transform=tfm)
        class_names = EMOTIONS
        print("Data mode: CSV")
    else:
        split_dir = os.path.join(args.data_dir, "Evaluate" if args.split == "test" else "train")
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Could not find split directory: {split_dir}. "
                "Either provide --csv or ensure --data-dir has train/ and Evaluate/."
            )
        ds = datasets.ImageFolder(split_dir, transform=tfm)
        class_names = ds.classes
        print(f"Data mode: folders ({args.data_dir})")
        print(f"Classes: {class_names}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    saved = 0
    sample_idx = 0

    for x, y in loader:
        x = x.to(device)
        with torch.enable_grad():
            logits, cams = cam_engine(x)

        preds = logits.argmax(dim=1).cpu().numpy()
        y_np = y.numpy()

        for i in range(x.size(0)):
            if saved >= args.num_images:
                break
            img_np = denorm_image(x[i])
            cam_np = cams[i]
            t = class_names[int(y_np[i])]
            pr = class_names[int(preds[i])]
            out_path = os.path.join(args.out_dir, f"sample_{sample_idx:05d}_{t}_pred_{pr}.png")
            save_gradcam_figure(img_np, cam_np, t, pr, out_path)
            saved += 1
            sample_idx += 1

        if saved >= args.num_images:
            break

    print(f"Saved {saved} Grad-CAM images to: {args.out_dir}")


if __name__ == "__main__":
    main()

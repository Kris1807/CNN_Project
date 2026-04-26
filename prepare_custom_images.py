import argparse
from pathlib import Path

from image_preprocessing import (
    build_export_transform,
    collect_image_paths,
    load_and_crop_image,
)


# Prepared images are saved with a predictable suffix so they are easy to match back to the source photo.
def build_output_name(image_path):
    return f"{Path(image_path).stem}_prepared.png"


def main():
    parser = argparse.ArgumentParser(
        description="Prepare custom images into the same visual format expected by the model."
    )
    parser.add_argument("--input-dir", help="Folder containing custom images")
    parser.add_argument("--images", nargs="*", help="Optional list of image paths")
    parser.add_argument("--out-dir", required=True, help="Folder to save prepared images")
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="resnet18")
    parser.add_argument(
        "--crop-mode",
        choices=["face", "full", "square", "portrait", "tight"],
        default="face",
        help="How to crop the image before formatting it for the model",
    )
    parser.add_argument("--min-images", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=None)
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

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Export preprocessing keeps the model's size/grayscale format without converting to tensors.
    export_transform = build_export_transform(args.model)

    print(f"Preparing {len(image_paths)} image(s) with crop mode: {args.crop_mode}")
    print(f"Saving outputs to: {output_dir}\n")

    for image_path in image_paths:
        _, cropped_image = load_and_crop_image(image_path, args.crop_mode)
        prepared_image = export_transform(cropped_image)
        output_path = output_dir / build_output_name(image_path)
        prepared_image.save(output_path)
        print(f"{image_path} -> {output_path}")


if __name__ == "__main__":
    main()

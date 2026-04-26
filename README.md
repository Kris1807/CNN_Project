# Kristian Pitshugin

# Dr. Nazish Tahir

# CSCI 6550

# Emotion Classification Project

This project classifies facial expressions into seven emotion classes:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

The project includes:

- model training
- final evaluation on a held-out FER set
- Grad-CAM visualizations
- prediction on personal images
- Grad-CAM on personal images
- live webcam prediction
- live webcam prediction with optional Grad-CAM

## The zip includes those files:

- `README.md`
- `requirements.txt`
- `best_resnet18.pt`
- `train.py`
- `evaluate.py`
- `grad_cam.py`
- `live_cam.py`
- `predict_custom_images.py`
- `prepare_custom_images.py`
- `custom_grad_cam.py`
- `models.py`
- `fer_dataset.py`
- `emotion_pipeline.py`
- `image_preprocessing.py`

## Expected Folder Names

This project uses these folder names:

- `train/` = FER training images
- `Evaluate/` = FER evaluation images
- `test/` = personal images for the custom-image workflow
- `test_prepared/` = generated grayscale crops for custom-image Grad-CAM

## Python Version

Python 3.10 or newer is recommended.

## Setup On A New Computer

1. Extract the zip.
2. Open a terminal inside the extracted project folder.
3. Create and activate a virtual environment.
4. Install the required packages.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Fastest Way To Run The Project

1. Demonstration is the live webcam mode:

```bash
python live_cam.py --weights best_resnet18.pt
```

2. To run live webcam prediction with Grad-CAM:

```bash
python live_cam.py --weights best_resnet18.pt --grad-cam
```

Useful webcam options:

- smoother labels: `--history 8`
- lighter Grad-CAM overlay: `--grad-cam-alpha 0.30`
- disable mirroring: `--no-mirror`

Example:

```bash
python live_cam.py --weights best_resnet18.pt --grad-cam  --history 8 --grad-cam-alpha 0.30
```

## Camera Permission Note

- On macOS, Terminal or the application running Python may need camera permission before the webcam opens.
- Live Grad-CAM is slower than normal live prediction because it computes an extra backward pass.

## Evaluate The Saved Model

This requires `Evaluate/` and `best_resnet18.pt`.

```bash
python evaluate.py --data-dir . --model resnet18 --weights best_resnet18.pt
```

## Generate Grad-CAM For FER Evaluation Images

This requires `Evaluate/` and `best_resnet18.pt`.

```bash
python grad_cam.py --data-dir . --model resnet18 --weights best_resnet18.pt --num-images 20 --out-dir gradcam_outputs
```

## Run The Model On Personal Images

Place at least 5 images inside the `test/` folder, then run:

```bash
python predict_custom_images.py --weights best_resnet18.pt --input-dir test --min-images 5 --crop-mode face --top-k 3
```

This prints the predicted emotion and confidence scores for each image.

## Prepare Personal Images Into Model-Style Crops

This creates grayscale prepared images in `test_prepared/`.

```bash
python prepare_custom_images.py --input-dir test --out-dir test_prepared --min-images 5 --crop-mode face
```

## Generate Grad-CAM For Personal Images

Using prepared personal images is the cleanest option:

```bash
python custom_grad_cam.py --weights best_resnet18.pt --input-dir test_prepared --min-images 5 --crop-mode full --out-dir custom_gradcam_outputs_prepared
```

You can also run custom Grad-CAM directly on the original personal images:

```bash
python custom_grad_cam.py --weights best_resnet18.pt --input-dir test --min-images 5 --crop-mode face --out-dir custom_gradcam_outputs
```

## Retrain The Model

This requires the `train/` folder. Images could be taken from Kaggle's webiste.

```bash
python train.py --data-dir . --model resnet18 --epochs 12 --out best_resnet18.pt
```

During training, one epoch means one full pass through the training split. After each epoch, the script also runs a validation pass and saves a new checkpoint if validation accuracy improves.

## Optional CSV Mode

The code also supports the original `fer2013.csv` format.
Images could be taken from Kaggle's webiste.

Train:

```bash
python train.py --csv /path/to/fer2013.csv --model resnet18 --epochs 12 --out best_resnet18.pt
```

Evaluate:

```bash
python evaluate.py --csv /path/to/fer2013.csv --model resnet18 --weights best_resnet18.pt
```

## Notes

- The pretrained checkpoint is the easiest way to demonstrate the project quickly.
- FER-style grayscale face images are the domain the model was trained on, so dataset performance is stronger than performance on arbitrary real-world selfies.
- The custom-image pipeline improves personal-image testing by applying face detection, cropping, grayscale conversion, and consistent resizing before inference.

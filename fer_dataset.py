import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

USAGE_MAP = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


# This wrapper reads the original FER2013 CSV format and converts each pixel string into an image piece.
class FER2013Dataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        self.transform = transform
        df = pd.read_csv(csv_path)
        usage = USAGE_MAP[split]
        self.df = df[df["Usage"] == usage].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ").reshape(48, 48)
        image = Image.fromarray(pixels, mode="L")
        label = int(row["emotion"])
        if self.transform:
            image = self.transform(image)
        return image, label

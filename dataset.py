import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class DetectionDataset(Dataset):
    def __init__(self, annot_path, transform=None):
        self.annot_path = annot_path
        self.transform = transform
        self.annotations = pd.read_csv(annot_path)

    def __getitem__(self, idx):
        annotation = self.annotations.iloc[idx]
        image_path = annotation["image_id"]
        image = Image.open(image_path).convert("RGB")
        boxes = torch.as_tensor(annotation["boxes"])
        labels = torch.as_tensor(annotation["labels"])

        target = {"boxes": boxes, "labels": labels}

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.annotations)

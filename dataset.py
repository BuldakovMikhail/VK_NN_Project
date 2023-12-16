import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


def normalizer(tensor):
    tensor = tensor.float()
    tensor /= 255

    return tensor


def denormalizer(tensor):
    tensor *= 255

    return tensor.type(torch.uint8)


class DetectionDataset(Dataset):
    def __init__(self, annot_path, transform=None):
        self.annot_path = annot_path
        self.transform = transform
        self.annotations = pd.read_pickle(annot_path)

    def __getitem__(self, idx):
        annotation = self.annotations.iloc[idx]
        image_path = annotation["image_id"]
        image = np.array(Image.open(image_path).convert("RGB"))
        boxes = torch.as_tensor(annotation["boxes"])
        boxes = torch.clip(boxes, min=0.0)
        labels = annotation["labels"]

        target = {"boxes": boxes, "labels": labels}

        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = torch.Tensor(transformed['labels']).int()

        image = normalizer(image)

        target = {"boxes": boxes, "labels": labels, 'image_id': torch.Tensor([idx]).int()}
        return image, target

    def __len__(self):
        return len(self.annotations)

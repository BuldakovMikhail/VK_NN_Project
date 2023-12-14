from PIL import Image
from torchvision import transforms
import torch
import seaborn as sns
import cv2

import numpy as np
import matplotlib.pyplot as plt


class FoodDetector(pl.LightningModule):
    def __init__(self, class_dict):
        self.model = retinanet_resnet50_fpn_v2(
            num_classes=len(class_dict), weights="DEFAULT"
        )

    def training_step(self, batch, batch_idx):
        images, target = batch

        losses = self.model(images, target)

        final_loss = sum([value for value in loss_dict.values()]).item()

        self.log("train_loss", final_loss, sync_dist=True)
        return final_loss

    def validation_step(self, batch, batch_idx):
        images, target = batch

        losses = self.model(images, target)

        final_loss = sum([value for value in loss_dict.values()]).item()

        self.log("validation_loss", final_loss, sync_dist=True)
        return final_loss

    def forward(self, images):
        if len(images.shape) == 4:
            preds = self.model(images)
        else:
            preds = self.model(images.unsqueeze(0))
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]

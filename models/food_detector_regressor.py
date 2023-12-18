import torch
from torch import nn
import lightning.pytorch as pl

from models.food_detector import FoodDetector
from config import config


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class FoodDetectorRegressor(pl.LightningModule):
    def __init__(self, class_dict, lr=2e-3):
        super().__init__()
        self.learning_rate = lr
        self.retina = FoodDetector(class_dict)
        self.retina.load_state_dict(torch.load(config["DETECTOR_WEIGHTS_PATH"]))
        self.backbone = self.retina.model.backbone

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(4096, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ELU(),
            torch.nn.Linear(1024, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 4)
        )

        self.regression_loss = RMSELoss()

        for param in self.backbone.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        images, target = batch
        cals = target['calories'].reshape(-1, 1)
        fats = target['fat'].reshape(-1, 1)
        carb = target['carb'].reshape(-1, 1)
        protein = target['protein'].reshape(-1, 1)
        target = torch.hstack((cals, fats, carb, protein)).type(torch.float32)

        embeddings = self.backbone(images)

        embed7 = embeddings['p7']
        print(embed7.shape)

        processed_embeddings = embed7.view(-1, 256 * 4 * 4)

        pred = self.regressor(processed_embeddings)
        loss = self.regression_loss(pred, target)

        self.log("train_loss", loss, sync_dist=True)

        return loss.type(torch.float32)

    def validation_step(self, batch, batch_idx):
        images, target = batch
        cals = target['calories'].reshape(-1, 1)
        fats = target['fat'].reshape(-1, 1)
        carb = target['carb'].reshape(-1, 1)
        protein = target['protein'].reshape(-1, 1)
        target = torch.hstack((cals, fats, carb, protein))

        #         with torch.no_grad():
        embeddings = self.backbone(images)

        embed7 = embeddings['p7']
        processed_embeddings = embed7.view(-1, 256 * 4 * 4)

        pred = self.regressor(processed_embeddings)
        loss = self.regression_loss(pred, target)

        self.log("validation_loss", loss, sync_dist=True)

        return loss

    def forward(self, images):
        if len(images.shape) == 4:
            preds = self.model(images)
        else:
            preds = self.model(images.unsqueeze(0))
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=self.learning_rate)
        return [optimizer]

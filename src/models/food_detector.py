import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2
import lightning.pytorch as pl
from torch.nn import MSELoss


class FoodDetector(pl.LightningModule):
    def __init__(self, class_dict, lr=3e-4):
        super().__init__()
        self.model = retinanet_resnet50_fpn_v2(num_classes=91, weights='DEFAULT')
        self.learning_rate = lr
        self.classLossFunc = lambda x, y: torch.sum(x != y)
        self.bboxLossFunc = MSELoss()

    def training_step(self, batch, batch_idx):
        images, target = batch

        loss_dict = self.model(images, target)

        final_loss = sum([value for value in loss_dict.values()])

        self.log("validation_loss", final_loss.item(), sync_dist=True)
        return final_loss

    def validation_step(self, batch, batch_idx):
        images, target = batch

        bboxes = [elem['boxes'] for elem in target]
        labels = [elem['labels'] for elem in target]

        predictions = self.model(images)
        pred_bboxes = [elem['boxes'] for elem in predictions]
        scores = [elem['scores'] for elem in predictions]
        pred_labels = [elem['labels'] for elem in predictions]

        bbox_loss = 0
        class_loss = 0
        for pred, scores, expected, pred_labels, expected_labels in zip(pred_bboxes, scores, bboxes, pred_labels,
                                                                        labels):
            idx = torchvision.ops.nms(pred, scores, 0.)[:len(expected)]

            if len(idx) == len(expected):
                bbox_loss += self.bboxLossFunc(pred[idx], expected)
                class_loss += self.classLossFunc(pred_labels[idx], expected_labels)

        final_loss = bbox_loss + class_loss

        self.log("validation_loss", final_loss.item(), sync_dist=True)
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

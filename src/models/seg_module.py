import multiprocessing as mp
from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from tqdm import tqdm

from src.utils.loss import TopKPixelCrossEntropyLoss
from src.utils.metrics import ClasswiseBatchAveragedDiceScore


class SegLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss: torch.nn.Module,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net, loss"])

        self.net = net

        self.criterion = loss
        self.dice = ClasswiseBatchAveragedDiceScore()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_loss_best = MinMetric()

        self.nc_scores = None
        self.nc_curves = None
        self.kandinsky_mask = None

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        self.val_loss.reset()

    def model_step(self, batch: Any):
        seg_targets = batch["target_segmentation"].long()

        x = batch["image"]

        out = self.forward(x)

        seg_logits = out["seg_logits"]

        seg_targets = seg_targets.float()
        loss = self.criterion(seg_logits, seg_targets)
        seg_probs = torch.sigmoid(seg_logits)

        return loss, seg_probs, seg_targets

    def training_step(self, batch: Any, batch_idx: int):
        loss, seg_probs, seg_targets = self.model_step(batch)

        self.train_loss(loss)
        dice_score = self.dice(seg_probs, seg_targets.long())

        self.log("train/loss", self.train_loss, prog_bar=True)

        for i in range(1, len(dice_score)):
            self.log(
                f"train/dice_fg{i}", dice_score[i], on_step=True, on_epoch=False, prog_bar=False
            )

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, seg_probs, seg_targets = self.model_step(batch)

        self.val_loss(loss)
        dice_score = self.dice(seg_probs, seg_targets.long())

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        for i in range(1, len(dice_score)):
            self.log(
                f"val/dice_fg{i}", dice_score[i], on_step=True, on_epoch=False, prog_bar=False
            )

        return seg_probs

    def test_step(self, batch: Any, batch_idx: int):
        _, seg_probs, seg_targets = self.model_step(batch)

        return seg_probs

    def on_load_checkpoint(self, checkpoint):
        if "nc_curves" in checkpoint:
            self.nc_curves = checkpoint["nc_curves"]
        if "kandinsky_mask" in checkpoint:
            self.kandinsky_mask = checkpoint["kandinsky_mask"]

    def configure_optimizers(self):
        params = self.parameters()

        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SegLitModule(None, None, None, None)

import torch
from lightning import LightningModule


class ClasswiseBatchAveragedDiceScore(LightningModule):
    def __init__(self):
        """Dice score for binary segmentation."""
        super().__init__()

    def forward(self, input, target):
        """Compute the Dice score per class, averaged over the batch.

        Args:
            input (torch.Tensor): Input tensor of shape (B, C, H, W)
            target (torch.Tensor): Target tensor of shape (B, C, H, W)
        """
        smooth = 1e-6
        # threshold the input to get the prediction
        pred = torch.where(input > 0.5, torch.ones_like(input), torch.zeros_like(input))

        intersection = (target * pred).sum(dim=(2, 3))
        union = (target + pred).sum(dim=(2, 3))

        dice = (2.0 * intersection + smooth) / (union + smooth)

        y_o = target.sum(dim=(2, 3))

        nans = torch.full_like(dice, float("nan"))
        dice = torch.where(y_o == 0, nans, dice)

        batch_avg_dice = dice.nanmean(dim=0)

        return batch_avg_dice

from typing import Optional, Union

import torch
import torch.nn as nn


class TopKPixelCrossEntropyLoss(nn.Module):
    def __init__(self, k_frac=0.1, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.k_frac = k_frac
        self.loss = nn.BCEWithLogitsLoss(reduction="none")

        if weight is not None:
            self.register_buffer("weight", torch.tensor(weight))
            self.weight: Optional[torch.Tensor]
        else:
            self.weight = None

    def forward(self, logits, targets):
        # logits: (B, C, H, W)

        loss = self.loss(logits, targets)  # (B, C, H, W)

        loss = loss.flatten(start_dim=2)  # (B, C, H * W)

        k = int(self.k_frac * loss.shape[2])
        loss, _ = torch.topk(loss, k, dim=2)  # (B, C, k)

        loss = loss.permute(1, 0, 2)  # (C, B, k)

        loss = loss.mean(dim=(1, 2))  # (C,)

        if self.weight is not None:
            return (loss * self.weight).mean()
        else:
            return loss.mean()

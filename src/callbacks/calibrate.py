from enum import Enum
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from lightning import Callback


class CalibrateCallback(Callback):
    def __init__(
        self,
        method: str = "pixel",
        output_dir: Union[str, None] = None,
        cluster_finder: Any = None,
        class_idx: Union[int, None] = None,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.method = method
        self.class_idx = class_idx
        if cluster_finder is not None:
            self.cluster_finder = cluster_finder

    def on_validation_epoch_start(self, trainer, pl_module):
        self.nc_scores = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is None:
            raise ValueError("No outputs from model")

        # lower the precision to save memory
        seg_probs = outputs.half()
        targets = batch["target_segmentation"].half()

        inverse_probs = 1 - seg_probs

        inverse_probs = torch.where(targets == 1, inverse_probs, torch.nan).cpu()

        self.nc_scores.append(inverse_probs)

    def on_validation_epoch_end(self, trainer, pl_module):
        nc_scores = torch.cat(self.nc_scores, dim=0).float()
        nc_curve_points = 100

        if self.method == "pixel":
            nc_curves = torch.nanquantile(nc_scores, torch.linspace(0, 1, nc_curve_points), dim=0)

            self.nc_curves = nc_curves
        elif self.method == "image":
            nc_scores_flat = nc_scores.permute(0, 2, 3, 1).flatten(0, 2)

            # we need to convert to numpy to use nanquantile for large tensors
            nc_scores_flat_np = nc_scores_flat.numpy()
            nc_curve_np = np.nanquantile(
                nc_scores_flat_np, np.linspace(0, 1, nc_curve_points), axis=0
            )

            nc_curve = torch.from_numpy(nc_curve_np)

            nc_curves = (
                nc_curve.unsqueeze(-1).unsqueeze(-1).expand(nc_curve_points, *nc_scores.shape[1:])
            )

            self.nc_curves = nc_curves
        elif self.method == "kandinsky":
            nc_curves = torch.nanquantile(nc_scores, torch.linspace(0, 1, nc_curve_points), dim=0)

            kandinsky_mask = self.cluster_finder.get_mask(nc_curves)

            # TODO simultaneously calibrate all classes
            cluster_labels = list(np.unique(kandinsky_mask))
            nc_scores_class = nc_scores[:, self.class_idx]
            for i in cluster_labels:
                nc_scores_component = nc_scores_class[:, kandinsky_mask == i].flatten().numpy()

                nc_curve_component = np.nanquantile(
                    nc_scores_component, np.linspace(0, 1, nc_curve_points)
                )
                nc_curve_component = (
                    torch.from_numpy(nc_curve_component)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(nc_curve_points, *nc_scores.shape[2:])
                    .float()
                )
                nc_curves[:, self.class_idx, kandinsky_mask == i] = nc_curve_component[
                    :, kandinsky_mask == i
                ]

            self.nc_curves = nc_curves

        else:
            raise ValueError(f"Unknown method {self.method}")

        if self.output_dir is not None:
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(trainer.default_root_dir)
        out_path = output_dir / "cmodel.ckpt"

        trainer.save_checkpoint(out_path)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["nc_curves"] = self.nc_curves

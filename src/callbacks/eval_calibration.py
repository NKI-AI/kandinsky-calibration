from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import relplot as rp
import seaborn as sns
import torch
from lightning import Callback
from tqdm import tqdm

foreground_classes = ["person"]


class EvalCalibrationCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_test_start(self, trainer, pl_module):
        self.epoch_preds = []
        self.epoch_targets = []

        if hasattr(pl_module, "nc_curves"):
            self.nc_curves = pl_module.nc_curves
        else:
            self.nc_curves = None

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is None:
            raise ValueError("No outputs from model")

        seg_probs = outputs.half()
        targets = batch["target_segmentation"] > 0.5

        self.epoch_preds.append(seg_probs.detach().cpu())
        self.epoch_targets.append(targets.detach().cpu())

    def calibration_metrics(self, preds, targets, root_dir):
        threshold = 0.02

        for c, class_name in enumerate(foreground_classes):
            preds_class = preds[:, c + 1]
            targets_class = targets[:, c + 1]

            # structure frequency
            num_fg = 1.0 * np.sum(targets_class, axis=0) / targets_class.shape[0]

            fig0, ax0 = plt.subplots(figsize=(10, 10))
            ax0.imshow(num_fg)
            fig0.tight_layout()
            plt.savefig(root_dir / "structure_frequency.png")

            eces = np.zeros(preds_class[0].shape)
            for i in tqdm(range(eces.shape[0])):
                for j in range(eces.shape[1]):
                    pixel_preds = preds_class[:, i, j]
                    pixel_targets = targets_class[:, i, j]
                    t_pixel_preds = pixel_preds[pixel_preds > threshold]
                    t_pixel_targets = pixel_targets[pixel_preds > threshold]
                    # smECE is slow for large datasets
                    # eces[i, j] = rp.smECE(t_pixel_preds, t_pixel_targets)
                    eces[i, j] = rp.metrics.binnedECE(t_pixel_preds, t_pixel_targets, nbins=20)

            # save pixelwise ece if needed
            # torch.save(eces, root_dir / f"{class_name}-pixelwise_ece.pt")

            ece_vmax = 0.3

            fig, ax = plt.subplots()
            im = ax.imshow(eces, cmap="viridis", vmin=0, vmax=ece_vmax)
            fig.colorbar(im, ax=ax, shrink=0.6)
            fig.tight_layout()
            plt.savefig(root_dir / f"{class_name}-pixelwise_ece.png", bbox_inches="tight")

            # imagewise reliability diagram
            t_preds = preds_class[preds_class > threshold]
            t_targets = targets_class[preds_class > threshold]

            fig2, ax2 = rp.rel_diagram_binned(
                t_preds,
                t_targets,
                nbins=20,
            )
            fig2.savefig(root_dir / f"{class_name}-rel_diagram_all.png", bbox_inches="tight")

            # midpixel reliability diagram
            midpixel_preds = preds_class[:, 120, 160]
            midpixel_targets = targets_class[:, 120, 160]
            t_midpixel_preds = midpixel_preds[midpixel_preds > threshold]
            t_midpixel_targets = midpixel_targets[midpixel_preds > threshold]
            fig3, ax3 = rp.rel_diagram_binned(
                t_midpixel_preds,
                t_midpixel_targets,
                nbins=20,
            )
            fig3.savefig(root_dir / f"{class_name}-rel_diagram_midpixel.png", bbox_inches="tight")

    def coverage_metrics(self, preds, targets, root_dir):
        self.set_default_style()
        ncs = 1 - preds

        for c, class_name in enumerate(foreground_classes):
            class_ncs = ncs[:, c + 1]
            class_targets = targets[:, c + 1]
            class_nc_curves = self.nc_curves[:, c + 1].numpy()

            steps = 20
            confidences = np.linspace(0, 1, steps + 1)
            coverages = np.zeros(steps)
            abs_errors = np.zeros(steps)
            for i in range(steps):
                c_mask = class_ncs <= class_nc_curves[int((i + 0.5) * 100 / steps) - 1]
                coverage = (class_targets * c_mask).sum() / class_targets.sum()
                coverages[i] = coverage
                abs_errors[i] = np.abs(coverage - (confidences[i + 1] + confidences[i]) / 2)

            fig1, ax1 = self.coverage_diagram(
                coverages, confidences, np.mean(abs_errors), f"class '{class_name}'"
            )
            plt.savefig(root_dir / f"{class_name}-coverage.png")
            marginal_out = {"coverages": coverages, "confidences": confidences}
            torch.save(marginal_out, root_dir / f"{class_name}-marginal.pt")

            px = 150
            py = 220
            class_midpixel_targets = class_targets[:, py, px]
            class_midpixel_ncs = class_ncs[:, py, px]
            class_midpixel_nc_curves = class_nc_curves[:, py, px]

            coverages_midpixel, confidences, abs_error_midpixel = self.pixel_coverage(
                class_midpixel_targets, class_midpixel_ncs, class_midpixel_nc_curves, steps
            )
            fig2, ax2 = self.coverage_diagram(
                coverages_midpixel,
                confidences,
                abs_error_midpixel,
                f"class '{class_name}' (midpixel)",
            )
            plt.savefig(root_dir / f"{class_name}-coverage-midpixel.png")

            ces = np.zeros(class_targets[0].shape)
            for i in tqdm(range(ces.shape[0])):
                for j in range(ces.shape[1]):
                    pixel_targets = class_targets[:, i, j]
                    # check if all class_targets are 0
                    if np.all(pixel_targets == 0):
                        ces[i, j] = np.nan
                        continue
                    pixel_ncs = class_ncs[:, i, j]
                    pixel_nc_curves = class_nc_curves[:, i, j]
                    coverages, confidences, abs_error = self.pixel_coverage(
                        pixel_targets, pixel_ncs, pixel_nc_curves, steps
                    )
                    ces[i, j] = abs_error

            torch.save(ces, root_dir / f"{class_name}-pixelwise_ce.pt")

            ce_vmax = 0.5

            fig3, ax3 = plt.subplots(figsize=(10, 10))
            im3 = ax3.imshow(ces, cmap="viridis", vmin=0, vmax=ce_vmax)
            # fig.colorbar(im3, ax=ax)
            fig3.tight_layout()
            plt.savefig(root_dir / f"{class_name}-pixelwise_ce.png")

    def pixel_coverage(self, targets, ncs, nc_curves, steps):
        confidences = np.linspace(0, 1, steps + 1)
        coverages = np.zeros(steps)
        abs_errors = np.zeros(steps)
        for i in range(steps):
            c_mask = ncs <= nc_curves[int((i + 1) * 100 / steps) - 1]
            coverage = (targets * c_mask).sum() / targets.sum()
            coverages[i] = coverage
            abs_errors[i] = np.abs(coverage - (confidences[i + 1] + confidences[i]) / 2)

        return coverages, confidences, np.mean(abs_errors)

    def on_test_epoch_end(self, trainer, pl_module):
        preds = torch.cat(self.epoch_preds, dim=0).numpy()
        targets = torch.cat(self.epoch_targets, dim=0).numpy()

        self.calibration_metrics(preds, targets, Path(trainer.default_root_dir))

        if self.nc_curves is not None:
            self.coverage_metrics(preds, targets, Path(trainer.default_root_dir))

    def coverage_diagram(self, coverages, confidences, mean_abs_error, title):
        steps = len(coverages)
        error_label = f"CE$_{{{steps}}}: {mean_abs_error:.3f}$"

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.stairs(coverages, confidences, color="red")
        ax.text(0.05, 0.9, error_label)
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")

        ax.set_xlabel("confidence")
        ax.set_ylabel("coverage")
        ax.set_title(title)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_aspect("equal")
        fig.tight_layout()
        return fig, ax

    def set_default_style(self):
        mpl.rc_file_defaults()
        sns.set_style("whitegrid")
        pal = sns.color_palette("pastel")
        sns.set_palette(pal, color_codes=True)

        mpl.rcParams.update(
            {
                "axes.edgecolor": "0.5",
                "font.size": 22,
                "legend.frameon": False,
                "patch.force_edgecolor": False,
                "figure.figsize": [6.0, 6.0],
                "axes.titlepad": 20,
            }
        )

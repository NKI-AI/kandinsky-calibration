import torch

from src.utils.pixel_clustering.genetic_optimization.objective_functions import (
    AnnulusObjectiveFunction,
)
from src.utils.pixel_clustering.genetic_optimization.transforms import (
    ConstrainToBounds,
    SortTrial,
    TransformFactory,
)


class GeneticAnnuliClusterFinder:
    def __init__(self, data, constrain_bounds, diff_evolution):
        self.data = data

        sort_trial = SortTrial(start_idx=2)
        constrain_to_bounds = ConstrainToBounds(bounds=constrain_bounds)
        self.transforms = TransformFactory(transforms=[sort_trial, constrain_to_bounds])
        self.diff_evolution = diff_evolution

    def get_mask(self, nc_curves):
        sampled_points = torch.stack([nc_curves[p] for p in self.data.curve_points], dim=0)
        sampled_points = sampled_points.permute(1, 2, 3, 0)
        sampled_height_width = sampled_points.shape[1:3]
        sampled_points = sampled_points[self.data.class_idx]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        objective_function = AnnulusObjectiveFunction(
            sampled_points=sampled_points,
            batch_size=self.data.batch_size,
            device=device,
        )

        diff_evolution = self.diff_evolution(
            objective_func=objective_function,
            post_transforms=self.transforms,
        )

        best_params, _ = diff_evolution.optimize()

        kandinsky_mask = self.create_annuli_mask(params=best_params, shape=sampled_height_width)

        return kandinsky_mask

    def create_annuli_mask(
        self, params: torch.Tensor, shape: torch.Size, **kwargs
    ) -> torch.Tensor:
        height, width = shape
        fixed_center_x, fixed_center_y = width // 2, height // 2

        y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
        coords = torch.stack((y, x), dim=-1)
        xy_coords = coords.view(-1, 2)
        xy_coords[:, 0] = (
            xy_coords[:, 0] * -1 + fixed_center_y
        )  # Flip y-axis to match image coordinates
        xy_coords[:, 1] -= fixed_center_x

        center_x, center_y = params[:2]
        radii = torch.cat((torch.tensor([0.0]), params[2:]))

        distances = (xy_coords[:, 1] - center_x) ** 2 + (xy_coords[:, 0] - center_y) ** 2

        mask = torch.zeros(height * width, dtype=torch.float32)

        for i, radius in enumerate(radii):
            mask[(distances > radii[i - 1] ** 2) & (distances <= radius**2)] = i

        mask[distances > radii[-1] ** 2] = len(radii)

        # Reshape the mask to the original image shape
        return mask.view(height, width)

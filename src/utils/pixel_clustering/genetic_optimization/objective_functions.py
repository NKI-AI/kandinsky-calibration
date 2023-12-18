import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class ObjectiveFunctionBase(ABC):
    @abstractmethod
    def __init__(self, sampled_points: Tensor, device: str, **kwargs):
        self._sampled_points = sampled_points.to(device)
        self._device = device
        pass

    @abstractmethod
    def __call__(self, parameters: torch.Tensor) -> float:
        """Evaluate the objective function given the parameters."""
        pass


class AnnulusObjectiveFunction(ObjectiveFunctionBase):
    def __init__(
        self, sampled_points: Tensor, device: str = "cpu", batch_size: Union[int, None] = None
    ):
        super().__init__(sampled_points, device)
        self._init_grid_and_cache()
        self._batch_size = batch_size

    def _init_grid_and_cache(self) -> None:
        self._H, self._W, self._M = self._sampled_points.shape
        self._n = self._H * self._W
        self._fixed_center_x, self._fixed_center_y = self._W // 2, self._H // 2
        self._xy_coords = self._generate_grid()

    def _generate_grid(self) -> Tensor:
        y, x = torch.meshgrid(torch.arange(self._H), torch.arange(self._W))
        coords = torch.stack((y, x), dim=-1)
        xy_coords = coords.view(-1, 2)
        xy_coords[:, 0] = (
            xy_coords[:, 0] * -1 + self._fixed_center_y
        )  # Flip y-axis to match image coordinates
        xy_coords[:, 1] -= self._fixed_center_x
        return xy_coords.to(self._device)

    def _find_annulus_indices(
        self, radii: Tensor, center_coords: Tuple[float, float]
    ) -> List[Tensor]:
        x_center, y_center = center_coords
        distances = (self._xy_coords[:, 1] - x_center) ** 2 + (
            self._xy_coords[:, 0] - y_center
        ) ** 2
        annulus_indices = [
            torch.where((radii[i] ** 2 <= distances) & (distances < radii[i + 1] ** 2))[0]
            for i in range(len(radii) - 1)
        ]
        annulus_indices.append(torch.where(radii[-1] ** 2 <= distances)[0])  # add the last annulus
        return annulus_indices

    def _evaluate_internal_distances(
        self, radii: Tensor, center_coords: Tuple[float, float]
    ) -> Tensor:
        radii = radii.to(self._device)

        # add 0.0 to the beginning of radii to include the center -- this is not part of optimization
        radii = torch.cat((torch.tensor([0.0]).to(self._device), radii))

        center_coords = tuple(coord.to(self._device) for coord in center_coords)
        annulus_indices = self._find_annulus_indices(radii, center_coords)
        points_Rm = self._sampled_points.reshape(-1, self._M)
        annulus_internal_distances = []

        if self._batch_size is not None:
            for indices in annulus_indices:
                annulus_points = points_Rm[indices]
                num_points = annulus_points.size(0)
                total_internal_distance = 0.0

                # Process in batches
                for start_index in range(0, num_points, self._batch_size):
                    end_index = min(start_index + self._batch_size, num_points)
                    batch = annulus_points[start_index:end_index]
                    distances = torch.cdist(batch, annulus_points, p=2.0)
                    total_internal_distance += distances.sum().item()  # aggregate the partial sums

                total_internal_distance *= 1 / self._n
                annulus_internal_distances.append(total_internal_distance)

        else:
            for indices in annulus_indices:
                annulus_points = points_Rm[indices]
                distances = torch.cdist(annulus_points, annulus_points, p=2.0)
                total_internal_distance = 1 / self._n * distances.sum().item()
                annulus_internal_distances.append(total_internal_distance)

        # logging.info(
        #     f"Total internal distances for each annulus: {annulus_internal_distances} and sum: {sum(annulus_internal_distances)}"
        # )
        # logging.info(f"Current radii: {radii} and center: {center_coords}")
        return torch.tensor(annulus_internal_distances).sum()

    def __call__(self, parameters: Tensor) -> Tensor:
        parameters = parameters.reshape(-1)
        center_x, center_y = parameters[0], parameters[1]
        radii = parameters[2:]
        return self._evaluate_internal_distances(radii, (center_x, center_y))

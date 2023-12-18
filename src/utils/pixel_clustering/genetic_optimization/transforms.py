import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# Abstract class for the transforms
class Transform(ABC):
    def __init__(self, **kwargs) -> None:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def __call__(self, data: Tensor) -> Any:
        pass


class SortTrial(Transform):
    def __init__(self, start_idx: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._start_idx = start_idx

    @property
    def name(self) -> str:
        return "SortTrial"

    def __call__(self, data: Tensor) -> Tensor:
        data[self._start_idx :] = torch.sort(data[self._start_idx :])[0]
        return data


class ConstrainToBounds(Transform):
    def __init__(self, bounds: list, **kwargs) -> None:
        super().__init__(**kwargs)
        bounds = torch.Tensor(bounds)
        self._lower_bounds, self._upper_bounds = bounds[:, 0], bounds[:, 1]

    @property
    def name(self) -> str:
        return "ConstrainToBounds"

    def __call__(self, data: Tensor) -> Tensor:
        return torch.max(torch.min(data, self._upper_bounds), self._lower_bounds)


class TransformFactory:
    def __init__(self, transforms: List[Any], **kwargs) -> None:
        self.transforms = []

        for transform in transforms:
            logger.info(f"Adding transform: {transform}")
            self.transforms.append(transform)

    def __call__(self, data: Tensor) -> Tensor:
        for transform in self.transforms:
            data = transform(data)
        return data

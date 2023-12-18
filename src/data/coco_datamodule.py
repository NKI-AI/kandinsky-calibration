import random
from pathlib import Path
from typing import Optional, Union

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.components.coco_dataset import CocoSegmentationDataset


class CocoDataModule(LightningDataModule):
    """Data module for CocoDetection dataset."""

    def __init__(
        self,
        trainval_root: Union[str, Path],
        test_root: Union[str, Path],
        trainval_split: str = "1000",
        batch_size: int = 16,
        num_workers: int = 0,
        dims: tuple = (256, 256),
        augmentations=None,
        cal_trim: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.trainval_root = Path(trainval_root)
        self.test_root = Path(test_root)

        self.trainval_split = trainval_split

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.cal_trim = cal_trim

        self._training_data: Optional[Dataset] = None
        self._val_data: Optional[Dataset] = None
        self._testing_data: Optional[Dataset] = None
        self.dims = dims
        self.augmentations = augmentations

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self._data_train = CocoSegmentationDataset(
                data_dir=self.trainval_root,
                manifest_fn=self.trainval_root
                / "splits"
                / f"t{self.trainval_split}"
                / "labels_train.json",
                dims=self.dims,
                transforms=self.augmentations,
            )
            self._data_val = CocoSegmentationDataset(
                data_dir=self.trainval_root,
                manifest_fn=self.trainval_root
                / "splits"
                / f"t{self.trainval_split}"
                / "labels_val.json",
                dims=self.dims,
            )

        elif stage == "validate":
            self._data_val = CocoSegmentationDataset(
                data_dir=self.trainval_root,
                manifest_fn=self.trainval_root
                / "splits"
                / f"t{self.trainval_split}"
                / "labels_val.json",
                dims=self.dims,
            )

        elif stage == "calibrate":
            ds = CocoSegmentationDataset(
                data_dir=self.trainval_root,
                manifest_fn=self.trainval_root
                / "splits"
                / f"t{self.trainval_split}"
                / "labels_cal.json",
                dims=self.dims,
            )

            if self.cal_trim is not None:
                n = self.cal_trim
                if n > len(ds):
                    n = len(ds)
                indices = list(range(len(ds)))
                random.seed(42)
                cal_inds = random.sample(indices, n)
                self._data_val = Subset(ds, cal_inds)
            else:
                self._data_val = ds

        elif stage == "test":
            self._data_test = CocoSegmentationDataset(
                data_dir=self.test_root,
                manifest_fn=self.test_root / "labels.json",
                dims=self.dims,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

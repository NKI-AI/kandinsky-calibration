import os

import fiftyone as fo
import fiftyone.zoo as foz

# this is where fiftyone will download the dataset
fo.config.dataset_zoo_dir = "<target fiftyone dataset zoo dir>"

# this is where fiftyone will export the subset containing only the person class
trainval_dir = "data/coco/trainval"
test_dir = "data/coco/test"

# We use the train split for training, validation, and calibration,
# and the validation split as test data
# since the COCO test split has no labels

data_trainval = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    dataset_name="coco_trainval_person",
    label_types=["segmentations"],
    classes=["person"],
)
data_trainval.export(
    export_dir=trainval_dir,
    dataset_type=fo.types.COCODetectionDataset,
)

data_test = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    dataset_name="coco_test_person",
    label_types=["segmentations"],
    classes=["person"],
)
data_test.export(
    export_dir=test_dir,
    dataset_type=fo.types.COCODetectionDataset,
)

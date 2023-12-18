import os
from pathlib import Path
from typing import Union

import PIL
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms as tf


class CocoSegmentationDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        manifest_fn: Union[str, Path],
        dims: tuple = (256, 256),
        transforms=None,
    ):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            data_dir: image directory.
            manifest_fn: coco annotation absolute file path.
        """
        self.data_dir = Path(data_dir) / "data"
        self.manifest_fn = Path(manifest_fn)
        self.coco = COCO(self.manifest_fn)
        self.ids = list(self.coco.imgs.keys())

        self.category_ids = [49]  # person
        self.categories = self.coco.loadCats(self.category_ids)
        self.dims = dims

        self.transform_image = None
        self.transform_mask = None
        if transforms is not None:
            transforms_image = transforms.copy()
            for i, t in enumerate(transforms_image):
                if hasattr(t, "interpolation"):
                    transforms_image[i].interpolation = tf.InterpolationMode.BILINEAR
            self.transform_image = tf.Compose(transforms_image)
            self.transform_mask = tf.Compose(transforms)

    def __getitem__(self, index):
        """Returns one data pair (image and annotation)."""

        id = self.ids[index]
        coco = self.coco
        img_path = coco.loadImgs(id)[0]["file_name"]
        img_pil = Image.open(os.path.join(self.data_dir, img_path)).convert("RGB")
        w, h = img_pil.size

        ann_ids = coco.getAnnIds(imgIds=id, catIds=self.category_ids, iscrowd=None)
        annotations = coco.loadAnns(ann_ids)

        # +1 for background
        mask = torch.zeros((h, w, len(self.category_ids) + 1), dtype=torch.long)

        for i, ann in enumerate(annotations):
            class_id = self.category_ids.index(ann["category_id"])
            if len(ann["segmentation"]) == 0:
                continue

            try:
                binary_mask = torch.from_numpy(coco.annToMask(ann))
            except Exception as e:
                raise e

            mask[:, :, class_id + 1] = torch.logical_or(mask[:, :, class_id + 1], binary_mask)

        # permute channel to be [C, H, W]
        mask = mask.permute(2, 0, 1)

        # resize and crop
        image, mask = self.resize_and_pad(img_pil, mask, self.dims)
        image = tf.ToTensor()(image)

        if self.transform_image is not None:
            state = torch.get_rng_state()
            image = self.transform_image(image)
            torch.set_rng_state(state)
            mask = mask.unsqueeze(0)
            mask = self.transform_mask(mask)
            mask = mask.squeeze(0)

        # background
        mask[0, :, :] = torch.logical_not(torch.any(mask[1:, :, :], dim=0))

        return {"image": image, "target_segmentation": mask}

    def __len__(self):
        return len(self.ids)

    def resize_and_pad(self, image: PIL.Image, mask: torch.Tensor, dims: tuple):
        w, h = image.size
        aspect_ratio = w / h

        if w > h:
            new_w = dims[0]
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = dims[1]
            new_w = int(new_h * aspect_ratio)

        mask = mask.unsqueeze(0)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = tf.functional.resize(mask, (new_h, new_w), tf.functional.InterpolationMode.NEAREST)

        pad_w = dims[0] - new_w
        pad_h = dims[1] - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        padded_image = tf.functional.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        padded_mask = tf.functional.pad(mask, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

        padded_mask = padded_mask.squeeze(0)

        return padded_image, padded_mask

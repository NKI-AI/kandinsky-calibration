import json
import random
from pathlib import Path

# the trainval split with images containing "person" contains ~64k images
n_train = 1000
n_val = 1000

coco_trainval_root = Path("data/coco/trainval")
out_dir = coco_trainval_root / f"splits/t{str(n_train)}"
out_dir.mkdir(exist_ok=True, parents=True)

print("loading manifest...")
with open(coco_trainval_root / "labels.json") as fl:
    labels = json.load(fl)

print(f"# images: {len(labels['images'])}")
print(f"# annotations: {len(labels['annotations'])}")

random.seed(42)

img_inds = range(len(labels["images"]))

print("sampling...")
train_inds = random.sample(img_inds, n_train)
rem_inds = list(set(img_inds) - set(train_inds))

val_inds = random.sample(rem_inds, n_val)

cal_inds = list(set(rem_inds) - set(val_inds))

print("done sampling")

# sanity check
print("Sanity check for overlap (should be empty)")
print(set(train_inds).intersection(set(val_inds)))
print(set(train_inds).intersection(set(cal_inds)))
print(set(val_inds).intersection(set(cal_inds)))

print("gathering images and annotations (slow)...")
train_imgs = [labels["images"][i] for i in train_inds]
val_imgs = [labels["images"][i] for i in val_inds]
cal_imgs = [labels["images"][i] for i in cal_inds]

train_ids = [i["id"] for i in train_imgs]
val_ids = [i["id"] for i in val_imgs]
cal_ids = [i["id"] for i in cal_imgs]

train_anns = [_ for _ in labels["annotations"] if _["image_id"] in train_ids]
val_anns = [_ for _ in labels["annotations"] if _["image_id"] in val_ids]
cal_anns = [_ for _ in labels["annotations"] if _["image_id"] in cal_ids]

print("done gathering")

labels["images"] = None
labels["annotations"] = None

labels_train = labels.copy()
labels_val = labels.copy()
labels_cal = labels.copy()

labels_train["images"] = train_imgs
labels_train["annotations"] = train_anns

labels_val["images"] = val_imgs
labels_val["annotations"] = val_anns

labels_cal["images"] = cal_imgs
labels_cal["annotations"] = cal_anns

print("writing...")

with open(out_dir / "labels_train.json", "w") as fl:
    json.dump(labels_train, fl)

with open(out_dir / "labels_val.json", "w") as fl:
    json.dump(labels_val, fl)

with open(out_dir / "labels_cal.json", "w") as fl:
    json.dump(labels_cal, fl)

"""
create_bdd_subset.py

Create a balanced 1,000-image subset of the BDD100K training set
that covers all 10 detection classes.
"""

import os
import json
import random
from collections import defaultdict, Counter
from shutil import copy2
from tqdm import tqdm

# -------------------------------------
# CONFIG
# -------------------------------------
SRC_JSON = "../bdd100k_labels_images_train.json"
SRC_IMG_DIR = "../bdd100k_images_100k/bdd100k/images/100k/train"
DST_IMG_DIR = "./bdd100k_subset_1k/images"
DST_JSON = "./bdd100k_subset_1k/labels.json"

NUM_IMAGES = 1000
CLASSES = [
    "car", "traffic sign", "traffic light", "person", "truck",
    "bus", "bike", "rider", "motor", "train"
]
EXCLUDE_CATS = {"lane", "drivable area"}
SEED = 42
random.seed(SEED)


# -------------------------------------
# STEP 1 — Load the full annotation file
# -------------------------------------
print("📂 Loading annotations...")
with open(SRC_JSON, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} image entries.")


# -------------------------------------
# STEP 2 — Build index: class -> images containing that class
# -------------------------------------
class_to_images = defaultdict(set)
image_to_labels = {}

for item in data:
    name = item["name"]
    labels = item.get("labels", [])
    cats = set()
    for lbl in labels:
        cat = lbl.get("category", "").lower()
        if cat in EXCLUDE_CATS:
            continue
        if cat in CLASSES:
            class_to_images[cat].add(name)
            cats.add(cat)
    if cats:
        image_to_labels[name] = item  # keep only annotated images

print(f"Number of usable images (with detection classes): {len(image_to_labels)}")


# -------------------------------------
# STEP 3 — Ensure coverage of all 10 classes
# -------------------------------------
subset_images = set()

# Start by picking at least one image per class
for cls in CLASSES:
    if class_to_images[cls]:
        subset_images.add(random.choice(list(class_to_images[cls])))

print(f"Selected {len(subset_images)} images (1 per class).")

# Fill up remaining slots to reach NUM_IMAGES
remaining_needed = NUM_IMAGES - len(subset_images)
if remaining_needed < 0:
    remaining_needed = 0

# Build a pool of all images (shuffle for randomness)
all_images = list(image_to_labels.keys())
random.shuffle(all_images)

for img in all_images:
    if img in subset_images:
        continue
    subset_images.add(img)
    if len(subset_images) >= NUM_IMAGES:
        break

subset_images = list(subset_images)
print(f"✅ Final subset size: {len(subset_images)} images.")


# -------------------------------------
# STEP 4 — Extract subset data
# -------------------------------------
subset_data = [image_to_labels[name] for name in subset_images]

# Compute per-class coverage stats
class_counter = Counter()
for item in subset_data:
    for lbl in item["labels"]:
        cat = lbl.get("category", "").lower()
        if cat in CLASSES:
            class_counter[cat] += 1

print("\n📊 Class distribution in subset:")
for cls in CLASSES:
    print(f" - {cls:<12}: {class_counter[cls]}")

# -------------------------------------
# STEP 5 — Copy images
# -------------------------------------
os.makedirs(DST_IMG_DIR, exist_ok=True)

print(f"\n📁 Copying images to {DST_IMG_DIR} ...")
for name in tqdm(subset_images):
    src_path = os.path.join(SRC_IMG_DIR, name)
    dst_path = os.path.join(DST_IMG_DIR, name)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(src_path):
        copy2(src_path, dst_path)
    else:
        print(f"⚠️ Missing file: {src_path}")

# -------------------------------------
# STEP 6 — Save subset JSON
# -------------------------------------
os.makedirs(os.path.dirname(DST_JSON), exist_ok=True)
with open(DST_JSON, "w") as f:
    json.dump(subset_data, f, indent=2)

print(f"\n✅ Saved subset JSON: {DST_JSON}")
print(f"📦 Subset complete! {len(subset_data)} images, covers {len(class_counter)} classes.")

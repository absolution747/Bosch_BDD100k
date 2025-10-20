import os
import json
import random
import shutil
from tqdm import tqdm
from PIL import Image

# ==========================
# CONFIGURATION
# ==========================
# Root paths
DATA_ROOT = "/workspace/data"
IMAGES_DIR = os.path.join(DATA_ROOT, "bdd100k_images_100k", "bdd100k", "images", "100k")
LABELS_DIR = os.path.join(DATA_ROOT, "bdd100k_labels_release", "bdd100k", "labels")

# Output subset path
OUTPUT_DIR = os.path.join(DATA_ROOT, "bdd100k_subset_yolo")

# Number of samples per split (adjust as needed)
NUM_TRAIN = 800
NUM_VAL = 200

# BDD100k class list
BDD_CLASSES = [
    "car", "traffic sign", "traffic light", "person", "truck",
    "bus", "bike", "rider", "motor", "train"
]


# ==========================
# HELPERS
# ==========================
def convert_bbox_to_yolo(img_w, img_h, x1, y1, x2, y2):
    """Convert bbox from (x1, y1, x2, y2) to normalized YOLO format."""
    x_center = ((x1 + x2) / 2.0) / img_w
    y_center = ((y1 + y2) / 2.0) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height


def convert_bdd_json_to_yolo(json_path, image_split, num_samples):
    """Convert BDD100k JSON annotations to YOLO format for a limited subset."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Randomly select subset
    subset = random.sample(data, num_samples)

    img_src_dir = os.path.join(IMAGES_DIR, image_split)
    img_dst_dir = os.path.join(OUTPUT_DIR, "images", image_split)
    lbl_dst_dir = os.path.join(OUTPUT_DIR, "labels", image_split)
    os.makedirs(img_dst_dir, exist_ok=True)
    os.makedirs(lbl_dst_dir, exist_ok=True)

    print(f"\n📦 Creating {image_split} subset ({num_samples} samples)...")

    for item in tqdm(subset, desc=f"Processing {image_split}"):
        img_name = item["name"]
        img_path = os.path.join(img_src_dir, img_name)

        if not os.path.exists(img_path):
            continue

        # Get image size
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            continue

        label_file = os.path.join(lbl_dst_dir, img_name.replace(".jpg", ".txt"))

        with open(label_file, "w") as lf:
            for obj in item.get("labels", []):
                if "box2d" not in obj:
                    continue
                category = obj["category"]
                if category not in BDD_CLASSES:
                    continue

                cls_id = BDD_CLASSES.index(category)
                box = obj["box2d"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                x, y, w, h = convert_bbox_to_yolo(img_w, img_h, x1, y1, x2, y2)
                lf.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        # Copy image
        shutil.copy(img_path, img_dst_dir)

    print(f"✅ Finished creating subset for {image_split}.")


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    random.seed(42)

    train_json = os.path.join(LABELS_DIR, "bdd100k_labels_images_train.json")
    val_json = os.path.join(LABELS_DIR, "bdd100k_labels_images_val.json")

    convert_bdd_json_to_yolo(train_json, "train", NUM_TRAIN)
    convert_bdd_json_to_yolo(val_json, "val", NUM_VAL)

    print("\n🎉 Subset created successfully!")
    print(f"Location: {OUTPUT_DIR}")

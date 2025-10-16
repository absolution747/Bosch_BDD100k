import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
from analyze import parse_bdd100k_labels

def bbox_wh_normalized(df, img_width=1280, img_height=720):
    """Compute normalized width & height of each bbox."""
    df = df.copy()
    df["w"] = (df["x2"] - df["x1"]) / img_width
    df["h"] = (df["y2"] - df["y1"]) / img_height
    df = df[(df["w"] > 0) & (df["h"] > 0)]
    return df[["w", "h", "category"]]

def iou(box, clusters):
    """Compute IoU between box and k cluster boxes (used as distance metric)."""
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection + 1e-8)
    return iou_

def avg_iou(boxes, clusters):
    return np.mean([np.max(iou(box, clusters)) for box in boxes])

def kmeans_iou(boxes, k=9, seed=42, max_iter=1000):
    """K-means clustering using IoU-based distance (for anchor box finding)."""
    np.random.seed(seed)
    # Randomly initialize cluster centers
    clusters = boxes[np.random.choice(len(boxes), k, replace=False)]
    for _ in range(max_iter):
        distances = np.array([1 - iou(box, clusters) for box in boxes])
        nearest = np.argmin(distances, axis=1)
        new_clusters = np.array([boxes[nearest == j].mean(axis=0) if np.any(nearest == j)
                                 else clusters[j] for j in range(k)])
        if np.allclose(clusters, new_clusters):
            break
        clusters = new_clusters
    return clusters

# ------------------------
# Main anchor computation
# ------------------------
def compute_yolo_anchors(df, k=9, img_width=1280, img_height=720):
    """
    Compute YOLO anchors (normalized w,h) using k-means IoU clustering.
    Returns cluster centers sorted by area (smallest→largest).
    """
    df = bbox_wh_normalized(df, img_width, img_height)
    boxes = df[["w", "h"]].values

    print(f"Running k-means IoU clustering with k={k} on {len(boxes)} boxes...")
    clusters = kmeans_iou(boxes, k=k)
    clusters = clusters[np.argsort(clusters[:, 0] * clusters[:, 1])]  # sort by area

    mean_iou = avg_iou(boxes, clusters)
    print(f"\n✅ Mean IoU between boxes and anchors: {mean_iou:.4f}")

    print("\n📦 YOLO Anchor Boxes (normalized to image size):")
    for i, (w, h) in enumerate(clusters):
        print(f" {i+1:2d}: ({w:.4f}, {h:.4f})")

    print("\nYOLO format (pixels for 1280×720):")
    for i, (w, h) in enumerate(clusters):
        print(f" {i+1:2d}: ({w*img_width:.1f}, {h*img_height:.1f})")

    # Plot cluster distribution
    plt.figure(figsize=(7, 6))
    plt.scatter(df["w"], df["h"], s=2, alpha=0.3, label="Boxes")
    plt.scatter(clusters[:, 0], clusters[:, 1], c="red", s=60, label="Anchors")
    plt.title(f"Anchor Box Clusters (k={k}) - Mean IoU: {mean_iou:.3f}")
    plt.xlabel("Width (normalized)")
    plt.ylabel("Height (normalized)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return clusters


if __name__ == "__main__":
    labels_path = "../bdd100k_labels_images_train.json"

    print("Parsing BDD100K annotations...")
    df = parse_bdd100k_labels(labels_path)

    print("Computing anchors for YOLOv8...")
    anchors = compute_yolo_anchors(df, k=9, img_width=1280, img_height=720)
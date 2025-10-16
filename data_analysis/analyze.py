import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import json
import pandas as pd
import seaborn as sns

def parse_bdd100k_labels(json_path):
    """
    Parse the BDD100K labels JSON file and exclude 'lane' and 'drivable area' entries.

    Args:
        json_path (str): Path to the BDD100K labels JSON file.

    Returns:
        pd.DataFrame: A DataFrame containing one row per bounding box annotation.
                      Columns: [image_name, category, x1, y1, x2, y2, occluded, truncated, weather, scene, timeofday]
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    parsed_records = []

    for item in tqdm(data, desc="Parsing BDD100K annotations"):
        image_name = item["name"]
        attributes = item.get("attributes", {})
        weather = attributes.get("weather", "undefined")
        scene = attributes.get("scene", "undefined")
        timeofday = attributes.get("timeofday", "undefined")

        for label in item.get("labels", []):
            category = label.get("category", "")
            if category in ["drivable area", "lane"]:
                continue  # exclude unwanted classes

            if "box2d" not in label:
                continue  # skip polygonal or segmentation labels

            box = label["box2d"]
            record = {
                "image_name": image_name,
                "category": category,
                "x1": box["x1"],
                "y1": box["y1"],
                "x2": box["x2"],
                "y2": box["y2"],
                "occluded": label.get("attributes", {}).get("occluded", False),
                "truncated": label.get("attributes", {}).get("truncated", False),
                "weather": weather,
                "scene": scene,
                "timeofday": timeofday,
            }
            parsed_records.append(record)

    df = pd.DataFrame(parsed_records)
    return df

def plot_bdd_piecharts(train_json, val_json):
    # Parse both training and validation data
    print("Loading training data...")
    train_df = parse_bdd100k_labels(train_json)

    print("Loading validation data...")
    val_df = parse_bdd100k_labels(val_json)

    # Attributes to plot
    attributes = ["weather", "scene", "timeofday"]

    # Set up the matplotlib figure
    fig, axes = plt.subplots(len(attributes), 2, figsize=(14, 14))
    fig.suptitle("BDD100K Dataset Distribution (Train vs Val)", fontsize=16, y=0.93)

    for i, attr in enumerate(attributes):
        # Compute counts
        train_counts = train_df[attr].value_counts()
        val_counts = val_df[attr].value_counts()

        # Plot Train pie chart
        axes[i, 0].pie(
            train_counts,
            labels=train_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 9},
        )
        axes[i, 0].set_title(f"Train Set - {attr.capitalize()}")

        # Plot Val pie chart
        axes[i, 1].pie(
            val_counts,
            labels=val_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 9},
        )
        axes[i, 1].set_title(f"Val Set - {attr.capitalize()}")

    plt.tight_layout()
    plt.show()

def plot_class_distribution_pie(train_json, val_json):
    # Load parsed data
    print("Loading training data...")
    train_df = parse_bdd100k_labels(train_json)

    print("Loading validation data...")
    val_df = parse_bdd100k_labels(val_json)

    # Get class counts
    train_counts = train_df["category"].value_counts()
    val_counts = val_df["category"].value_counts()

    # Align both distributions so classes match
    all_classes = sorted(set(train_counts.index) | set(val_counts.index))
    train_counts = train_counts.reindex(all_classes, fill_value=0)
    val_counts = val_counts.reindex(all_classes, fill_value=0)

    # Plot side-by-side pie charts
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Training Set Pie Chart
    wedges1, texts1, autotexts1 = axes[0].pie(
        train_counts,
        labels=all_classes,
        autopct=lambda p: f'{p:.1f}%' if p > 1 else '',
        startangle=90,
        counterclock=False
    )
    axes[0].set_title("Class Distribution - Training Set", fontsize=14)

    # Validation Set Pie Chart
    wedges2, texts2, autotexts2 = axes[1].pie(
        val_counts,
        labels=all_classes,
        autopct=lambda p: f'{p:.1f}%' if p > 1 else '',
        startangle=90,
        counterclock=False
    )
    axes[1].set_title("Class Distribution - Validation Set", fontsize=14)

    # Improve layout
    plt.tight_layout()
    plt.show()

    return train_counts, val_counts


def plot_annotations_per_image(train_json, val_json):
    print("Loading training data...")
    train_df = parse_bdd100k_labels(train_json)
    print("Loading validation data...")
    val_df = parse_bdd100k_labels(val_json)

    # Ensure correct column name
    image_col = "image_name" if "image_name" in train_df.columns else "image"

    # Count number of annotations per image
    train_counts = train_df.groupby(image_col).size()
    val_counts = val_df.groupby(image_col).size()

    # Compute basic statistics
    def get_stats(series):
        return {
            "mean": np.mean(series),
            "median": np.median(series),
            "90th %ile": np.percentile(series, 90),
            "max": np.max(series),
            "min": np.min(series),
            "count": len(series)
        }

    train_stats = get_stats(train_counts)
    val_stats = get_stats(val_counts)

    # Convert to DataFrame for tabular display
    stats_df = pd.DataFrame([train_stats, val_stats], index=["Train", "Validation"])

    # Plot histograms
    plt.figure(figsize=(14, 6))
    bins = np.arange(0, max(train_counts.max(), val_counts.max()) + 2, 2)

    plt.hist(train_counts, bins=bins, alpha=0.6, label="Train", edgecolor='black', color='royalblue')
    plt.hist(val_counts, bins=bins, alpha=0.6, label="Validation", edgecolor='black', color='darkorange')

    plt.xlabel("Number of Annotations per Image", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.title("Distribution of Annotations per Image (Train vs Validation)", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print the statistics table
    print("\n📊 Annotation Count Summary per Image:")
    print(stats_df.to_markdown())

    return train_counts, val_counts, stats_df

def compare_mean_relative_bbox_size(train_json, val_json, image_width=1280, image_height=720):
    """
    Compute and compare mean relative bounding box size per class
    between training and validation sets in BDD100K.
    """

    print("Loading training data...")
    train_df = parse_bdd100k_labels(train_json)

    print("Loading validation data...")
    val_df = parse_bdd100k_labels(val_json)

    # Compute relative bbox area
    def compute_relative_area(df):
        df = df.copy()
        df["width"] = df["x2"] - df["x1"]
        df["height"] = df["y2"] - df["y1"]
        df["rel_area"] = (df["width"] * df["height"]) / (image_width * image_height)
        df = df[df["rel_area"] > 0]
        return df

    train_df = compute_relative_area(train_df)
    val_df = compute_relative_area(val_df)

    # Compute mean relative area per class
    train_means = train_df.groupby("category")["rel_area"].mean().sort_index()
    val_means = val_df.groupby("category")["rel_area"].mean().sort_index()

    # Combine into a single DataFrame
    compare_df = pd.DataFrame({
        "Train Mean Rel Size": train_means,
        "Val Mean Rel Size": val_means
    }).fillna(0)

    # Print in formatted style
    print("\n📏 Mean Relative Size per Class (Train vs Validation):")
    print("----------------------------------------------------")

    for cls in compare_df.index:
        print(f" - {cls:15s}: Train={compare_df.loc[cls, 'Train Mean Rel Size']:.6f} | Val={compare_df.loc[cls, 'Val Mean Rel Size']:.6f}")

    # Optionally, return the DataFrame for further analysis
    return compare_df

def plot_aspect_ratio_distribution(train_json, val_json, bins=50):
    """
    Compare aspect ratio (width/height) distributions for each class
    between training and validation datasets with large, readable plots.
    """

    print("Loading and parsing training data...")
    train_df = parse_bdd100k_labels(train_json)

    print("Loading and parsing validation data...")
    val_df = parse_bdd100k_labels(val_json)

    # --- Compute aspect ratios ---
    def add_aspect_ratio(df):
        df = df.copy()
        df["width"] = df["x2"] - df["x1"]
        df["height"] = df["y2"] - df["y1"]
        df = df[(df["width"] > 0) & (df["height"] > 0)]
        df["aspect_ratio"] = df["width"] / df["height"]
        return df

    train_df = add_aspect_ratio(train_df)
    val_df = add_aspect_ratio(val_df)

    # --- Summarize per class ---
    def summarize(df):
        summary = {}
        for cls, grp in df.groupby("category"):
            ratios = grp["aspect_ratio"]
            summary[cls] = {
                "mean": np.mean(ratios),
                "median": np.median(ratios),
                "var": np.var(ratios),
                "p10": np.percentile(ratios, 10),
                "p90": np.percentile(ratios, 90),
                "n": len(ratios),
            }
        return summary

    train_summary = summarize(train_df)
    val_summary = summarize(val_df)

    # --- Print formatted summaries ---
    print("\nAspect Ratio Summary per Class (Train):")
    print("----------------------------------")
    for cls, stats in sorted(train_summary.items()):
        print(f"{cls:15s}: mean={stats['mean']:.2f}, median={stats['median']:.2f}, "
              f"var={stats['var']:.2f}, 10–90%={stats['p10']:.2f}–{stats['p90']:.2f}, n={stats['n']}")

    print("\nAspect Ratio Summary per Class (Validation):")
    print("----------------------------------")
    for cls, stats in sorted(val_summary.items()):
        print(f"{cls:15s}: mean={stats['mean']:.2f}, median={stats['median']:.2f}, "
              f"var={stats['var']:.2f}, 10–90%={stats['p10']:.2f}–{stats['p90']:.2f}, n={stats['n']}")

    # --- Plot histograms (one class per row) ---
    common_classes = sorted(set(train_df["category"]) & set(val_df["category"]))
    n_classes = len(common_classes)

    fig, axes = plt.subplots(nrows=n_classes, ncols=1, figsize=(14, n_classes * 3.5))
    if n_classes == 1:
        axes = [axes]  # handle single-class case

    for ax, cls in zip(axes, common_classes):
        train_aspect = train_df[train_df["category"] == cls]["aspect_ratio"]
        val_aspect = val_df[val_df["category"] == cls]["aspect_ratio"]

        ax.hist(
            train_aspect,
            bins=bins,
            alpha=0.6,
            label="Train",
            color="royalblue",
            edgecolor="black",
            density=True,
        )
        ax.hist(
            val_aspect,
            bins=bins,
            alpha=0.6,
            label="Validation",
            color="darkorange",
            edgecolor="black",
            density=True,
        )

        ax.set_title(f"{cls}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Aspect Ratio (width / height)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.suptitle("Aspect Ratio Distribution per Class (Train vs Validation)", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()

    return train_summary, val_summary

def compute_iou(box1, box2):
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area


def compute_mean_iou_matrix(df, iou_thresh=0.5):
    """
    Compute mean IoU overlaps per class pair (IoU ≥ threshold).

    Args:
        df (pd.DataFrame): parsed BDD100K annotations
        iou_thresh (float): IoU threshold (default 0.5)
    """
    print("Computing IoU overlaps across all images...")

    classes = sorted(df["category"].unique())
    n_classes = len(classes)
    iou_sum = np.zeros((n_classes, n_classes))
    count = np.zeros((n_classes, n_classes))

    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Process each image separately
    for image_id, group in tqdm(df.groupby("image_name")):
        boxes = group[["x1", "y1", "x2", "y2"]].values
        labels = group["category"].values

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = compute_iou(boxes[i], boxes[j])
                if iou >= iou_thresh:
                    ci, cj = class_to_idx[labels[i]], class_to_idx[labels[j]]
                    iou_sum[ci, cj] += iou
                    iou_sum[cj, ci] += iou  # symmetric
                    count[ci, cj] += 1
                    count[cj, ci] += 1

    # Avoid divide-by-zero
    mean_iou = np.divide(iou_sum, count, out=np.zeros_like(iou_sum), where=count > 0)
    return pd.DataFrame(mean_iou, index=classes, columns=classes), count


def plot_iou_heatmap(iou_matrix):
    """Visualize class–class IoU overlaps as a heatmap."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        iou_matrix,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Mean IoU (≥ 0.5)"},
    )
    plt.title("Mean IoU Overlaps per Class Pair (IoU ≥ 0.5)", fontsize=14, fontweight="bold")
    plt.xlabel("Class B")
    plt.ylabel("Class A")
    plt.tight_layout()
    plt.show()

def compute_cooccurrence_matrix(df):
    """
    Compute inter-class co-occurrence matrix for a parsed BDD100K dataset.

    Args:
        df (pd.DataFrame): Parsed dataset with columns ['image_name', 'category'].

    Returns:
        co_matrix (pd.DataFrame): Normalized co-occurrence matrix (0–1).
    """
    classes = sorted(df["category"].unique())
    n_classes = len(classes)
    co_matrix = np.zeros((n_classes, n_classes), dtype=np.float64)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    print(f"Computing co-occurrence matrix across {len(df['image_name'].unique())} images...")

    for image_id, group in tqdm(df.groupby("image_name")):
        unique_classes = group["category"].unique()
        indices = [class_to_idx[c] for c in unique_classes]
        for i in indices:
            for j in indices:
                co_matrix[i, j] += 1  # mark class co-occurrence

    # Normalize by number of images → frequency in [0, 1]
    total_images = len(df["image_name"].unique())
    co_matrix = co_matrix / total_images

    return pd.DataFrame(co_matrix, index=classes, columns=classes)


def plot_cooccurrence_heatmap(co_matrix):
    """Visualize inter-class co-occurrence as a heatmap."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        co_matrix,
        cmap="YlGnBu",
        annot=False,
        linewidths=0.5,
        cbar_kws={"label": "Fraction of Images with Both Classes"},
    )
    plt.title("Inter-class Co-occurrence Matrix (Train Set)", fontsize=16, fontweight="bold")
    plt.xlabel("Class B")
    plt.ylabel("Class A")
    plt.tight_layout()
    plt.show()


def summarize_cooccurrence(co_matrix):
    """Print most and least frequent co-occurring class pairs."""
    flat = co_matrix.stack().reset_index()
    flat.columns = ["Class A", "Class B", "Co-occurrence"]

    # Remove diagonal (self-co-occurrence)
    flat = flat[flat["Class A"] != flat["Class B"]]

    print("\n🔝 Top 10 Most Frequent Class Pairs:")
    print(flat.nlargest(10, "Co-occurrence").to_string(index=False))

    print("\n🔻 10 Least Frequent Class Pairs:")
    print(flat.nsmallest(10, "Co-occurrence").to_string(index=False))

from sklearn.cluster import KMeans

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
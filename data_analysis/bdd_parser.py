import json
import pandas as pd
from tqdm import tqdm

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


if __name__ == "__main__":
    # Example usage
    labels_path = "bdd100k_labels_images_train.json"  # update this path
    df = parse_bdd100k_labels(labels_path)
    print(df.head())
    print(f"\nParsed {len(df)} annotations across {df['image_name'].nunique()} images.")

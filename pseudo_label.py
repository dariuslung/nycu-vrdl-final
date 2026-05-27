import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

def mask_to_polygons(mask):
    """
    Converts a 2D binary mask into a list of polygons compatible with HuBMAP JSONL format.
    Uses contour approximation to reduce JSON file bloat and memory usage.
    """
    # Find external contours (ignore holes inside vessels)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    
    for contour in contours:
        if contour.shape[0] >= 3:
            # Approximate the polygon to smooth edges and reduce point count
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if approx.shape[0] >= 3:
                polygons.append(approx.squeeze(1).tolist())
                
    return polygons

def main():
    parser = argparse.ArgumentParser(description="Generate Pseudo-Labels for HuBMAP Datasets 2 & 3")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained Stage 1 (Teacher) weights")
    parser.add_argument("--conf", type=float, default=0.60, help="Confidence threshold for pseudo-labels")
    args = parser.parse_args()

    # Paths
    raw_img_dir = "data/train"
    orig_jsonl_path = "data/polygons.jsonl"
    meta_csv_path = "data/tile_meta.csv"
    output_jsonl_path = "data/pseudo_polygons.jsonl"
    
    print("Loading metadata...")
    meta_df = pd.read_csv(meta_csv_path)
    
    # Isolate Datasets
    ds1_ids = set(meta_df[meta_df['dataset'] == 1]['id'].astype(str))
    ds23_ids = set(meta_df[meta_df['dataset'].isin([2, 3])]['id'].astype(str))

    print(f"Found {len(ds1_ids)} Dataset 1 tiles and {len(ds23_ids)} Dataset 2/3 tiles.")
    
    # 1. Copy over the highly curated Dataset 1 Truths
    print("Migrating Dataset 1 Ground Truths...")
    with open(orig_jsonl_path, "r") as f, open(output_jsonl_path, "w") as out_f:
        for line in f:
            data = json.loads(line)
            if data['id'] in ds1_ids:
                out_f.write(line)
                
    # 2. Generate Pseudo-labels for Dataset 2 & 3
    print(f"Loading Teacher Model from {args.weights}...")
    model = YOLO(args.weights)
    class_names = {0: "blood_vessel", 1: "glomerulus"}
    
    print("Starting pseudo-label inference...")
    with open(output_jsonl_path, "a") as out_f:
        for i, img_id in enumerate(ds23_ids):
            img_path = os.path.join(raw_img_dir, f"{img_id}.tif")
            if not os.path.exists(img_path):
                continue
                
            # Read native image
            img = cv2.imread(img_path)
            
            # retina_masks=True ensures masks are scaled exactly to the native input image shape
            results = model.predict(img, imgsz=1024, conf=args.conf, retina_masks=True, verbose=False)[0]
            
            annotations = []
            if results.masks is not None:
                masks = results.masks.data.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                
                for mask, cls_id in zip(masks, classes):
                    cls_id = int(cls_id)
                    if cls_id not in class_names:
                        continue # We skip "unsure" for pseudo-labeling
                        
                    polygons = mask_to_polygons(mask)
                    if polygons:
                        # Append each disconnected mask instance as a distinct annotation
                        annotations.append({
                            "type": class_names[cls_id],
                            "coordinates": polygons
                        })
            
            out_f.write(json.dumps({"id": img_id, "annotations": annotations}) + "\n")
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(ds23_ids)} tiles...")
                
    print(f"Complete. Combined dataset saved to {output_jsonl_path}")

if __name__ == "__main__":
    main()
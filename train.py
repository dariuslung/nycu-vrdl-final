import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from ultralytics import settings
import yaml
import cv2


def prepare_yolo_dataset(raw_img_dir, jsonl_path, meta_csv_path, yolo_base_dir, img_size=1024, dataset1_only=False):
    print(f"Preparing YOLO dataset structure (Dataset 1 Only: {dataset1_only})...")
    
    yolo_base = Path(yolo_base_dir)
    dirs = {
        'train_img': yolo_base / 'images' / 'train',
        'val_img': yolo_base / 'images' / 'val',
        'train_lbl': yolo_base / 'labels' / 'train',
        'val_lbl': yolo_base / 'labels' / 'val'
    }
    
    # Create directories if they don't exist
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    class_mapping = {
        "blood_vessel": 0,
        "glomerulus": 1,
        "unsure": 2
    }
    
    # Parse JSONL
    annotations_dict = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                annotations_dict[data['id']] = data['annotations']
    else:
        raise FileNotFoundError(f"Annotation file not found at {jsonl_path}")

    # Gather all images
    all_images = [f for f in os.listdir(raw_img_dir) if f.endswith('.tif')]
    
    # Filter for Dataset 1 if requested
    if dataset1_only and os.path.exists(meta_csv_path):
        meta_df = pd.read_csv(meta_csv_path)
        valid_ids = set(meta_df[meta_df['dataset'] == 1]['id'].astype(str))
        all_images = [img for img in all_images if os.path.splitext(img)[0] in valid_ids]

    train_imgs, val_imgs = train_test_split(all_images, test_size=0.1, random_state=42)
    
    def process_split(img_list, split_name):
        img_dir = dirs[f'{split_name}_img']
        lbl_dir = dirs[f'{split_name}_lbl']
        
        for img_name in img_list:
            img_id = os.path.splitext(img_name)[0]
            src_img_path = os.path.join(raw_img_dir, img_name)
            dst_img_path = img_dir / img_name
            
            # Read native dimensions
            img = cv2.imread(src_img_path)
            if img is None:
                continue
            native_h, native_w = img.shape[:2] 
            
            if not dst_img_path.exists():
                shutil.copy(src_img_path, dst_img_path)
                
            # Create label file
            label_file = lbl_dir / f"{img_id}.txt"
            annos = annotations_dict.get(img_id, [])
            
            with open(label_file, 'w') as lf:
                for anno in annos:
                    class_name = anno.get('type')
                    class_id = class_mapping.get(class_name)
                    
                    if class_id is None or not anno.get('coordinates'):
                        continue
                        
                    for poly in anno['coordinates']:
                        poly_np = np.array(poly, dtype=np.float32)
                        
                        # Normalize based on NATIVE image size, not target training size
                        poly_np[:, 0] /= native_w
                        poly_np[:, 1] /= native_h
                        
                        # Ensure coordinates are strictly clipped between 0 and 1
                        poly_np = np.clip(poly_np, 0.0, 1.0)
                        
                        flat_coords = poly_np.flatten().tolist()
                        coords_str = " ".join([f"{c:.6f}" for c in flat_coords])
                        
                        lf.write(f"{class_id} {coords_str}\n")

    process_split(train_imgs, 'train')
    process_split(val_imgs, 'val')
    print("Dataset preparation complete.")


def create_yaml_config(yolo_base_dir, yaml_path):
    """Generates the data.yaml file required by YOLO."""
    config = {
        'path': os.path.abspath(yolo_base_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'blood_vessel',
            1: 'glomerulus',
            2: 'unsure'
        }
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return os.path.abspath(yaml_path)


def main():
    parser = argparse.ArgumentParser(description="Train HuBMAP YOLO-X Segmentation Model")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the training run")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1, help="1: All data, 2: Dataset 1 only")
    parser.add_argument("--weights", type=str, default="yolo11x-seg.pt", help="Path to initial weights")
    args = parser.parse_args()

    # Force update settings to ensure TensorBoard is active
    settings.update({'tensorboard': True})

    # Paths
    raw_img_dir = "data/train"
    jsonl_path = "data/polygons.jsonl"
    meta_csv_path = "data/tile_meta.csv"
    yolo_base_dir = f"datasets/hubmap_stage{args.stage}"
    yaml_path = f"datasets/hubmap_stage{args.stage}/data.yaml"
    
    dataset1_only = (args.stage == 2)
    
    if not os.path.exists(yaml_path):
        prepare_yolo_dataset(raw_img_dir, jsonl_path, meta_csv_path, yolo_base_dir, img_size=1024, dataset1_only=dataset1_only)
        create_yaml_config(yolo_base_dir, yaml_path)

    model = YOLO(args.weights)
    epochs = 100 if args.stage == 1 else 30

    print(f"Starting Stage {args.stage} training for run: {args.run_name}")
    results = model.train(
        data=yaml_path,
        project=os.path.abspath("outputs"),
        name=args.run_name,
        
        # --- COMPUTE & HARDWARE ---
        device=0,
        imgsz=1024,       # Upscaled resolution
        batch=8,          # Adjusted for 24GB VRAM at 1024x1024
        workers=8,

        # --- TRAINING SCHEDULE ---
        epochs=epochs,
        patience=20,
        optimizer="AdamW", 
        lr0=0.001 if args.stage == 1 else 0.0001,  # Lower LR for fine-tuning
        lrf=0.01,         # Final LR fraction
        weight_decay=0.05,# Aggressive weight decay to prevent overfitting the X model
        cos_lr=True,      # Cosine learning rate scheduler
        warmup_epochs=3 if args.stage == 1 else 0,
        
        # --- ADVANCED AUGMENTATIONS (Histology Tuned) ---
        hsv_h=0.02,       # Slight hue shifts (stain variance)
        hsv_s=0.3,        # Saturation variance
        hsv_v=0.3,        # Brightness variance
        degrees=45.0,     # Tissues have no definitive "up", full rotation is safe
        translate=0.2,
        scale=0.3,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.25,      # Re-introducing mild mosaic to help with edge truncation
        mixup=0.1,        # Slight mixup for regularization
        
        # --- LOSS WEIGHTS ---
        box=7.5,          # Prioritize bounding box accuracy
        cls=0.5,
        dfl=1.5,          # Distribution Focal Loss for finer bounding box edges
    )
    
    print("Training Complete. Results saved to:", results.save_dir)


if __name__ == "__main__":
    main()
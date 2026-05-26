import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from ultralytics import settings
import yaml


def prepare_yolo_dataset(raw_img_dir, jsonl_path, yolo_base_dir, img_size=512):
    """
    Parses HuBMAP polygons.jsonl and converts it into the YOLO segmentation format.
    YOLO format requires normalized polygon coordinates: class x1 y1 x2 y2 ...
    """
    print("Preparing YOLO dataset structure...")
    
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
    
    # Train/Val Split (90/10)
    train_imgs, val_imgs = train_test_split(all_images, test_size=0.1, random_state=42)
    
    def process_split(img_list, split_name):
        img_dir = dirs[f'{split_name}_img']
        lbl_dir = dirs[f'{split_name}_lbl']
        
        for img_name in img_list:
            img_id = os.path.splitext(img_name)[0]
            src_img_path = os.path.join(raw_img_dir, img_name)
            dst_img_path = img_dir / img_name
            
            # Copy image
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
                        
                        # Normalize coordinates to 0.0 - 1.0 based on image size
                        poly_np[:, 0] /= img_size
                        poly_np[:, 1] /= img_size
                        
                        # YOLO expects a flat list: x1 y1 x2 y2 ...
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
    parser = argparse.ArgumentParser(description="Train HuBMAP YOLO Segmentation Model")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the training run")
    args = parser.parse_args()

    # Force update settings to ensure TensorBoard is active
    settings.update({'tensorboard': True})

    # Paths
    raw_img_dir = "data/train"
    jsonl_path = "data/polygons.jsonl"
    yolo_base_dir = "datasets/hubmap"
    yaml_path = "datasets/hubmap/data.yaml"
    
    # 1. Convert dataset to YOLO format (Runs once)
    if not os.path.exists(yaml_path):
        prepare_yolo_dataset(raw_img_dir, jsonl_path, yolo_base_dir)
        create_yaml_config(yolo_base_dir, yaml_path)
    else:
        print("YOLO dataset format already exists. Skipping preparation.")

    # 2. Initialize YOLO model
    # 'yolo11m-seg.pt' handles the accuracy/speed tradeoff perfectly for an 8GB GPU.
    model = YOLO("yolo11m-seg.pt")

    # 3. Train the model
    print(f"Starting training for run: {args.run_name}")
    results = model.train(
        data=yaml_path,
        project=os.path.abspath("outputs"), # Forces the exact directory
        name=args.run_name,
        epochs=30,
        imgsz=512,        # Native HuBMAP tile size
        batch=8,          # Fits well within 8GB VRAM
        device=0,         # Uses first CUDA GPU
        workers=4,
        optimizer="SGD",
        lr0=0.01,
        weight_decay=0.0001,
        # Augmentations tuned for histology
        hsv_h=0.015,
        hsv_s=0.2,
        hsv_v=0.2,
        degrees=15.0,
        translate=0.1,
        scale=0.2,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.0,       # Disabled; mosaic can distort cellular boundaries
        mixup=0.0
    )
    
    print("Training Complete. Results saved to:", results.save_dir)


if __name__ == "__main__":
    main()
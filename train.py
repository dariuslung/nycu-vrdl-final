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


def prepare_yolo_dataset(raw_img_dir, jsonl_path, meta_csv_path, yolo_base_dir, stage=1):
    print(f"Preparing YOLO dataset structure for Stage {stage}...")
    
    yolo_base = Path(yolo_base_dir)
    dirs = {
        'train_img': yolo_base / 'images' / 'train',
        'val_img': yolo_base / 'images' / 'val',
        'train_lbl': yolo_base / 'labels' / 'train',
        'val_lbl': yolo_base / 'labels' / 'val'
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    class_mapping = {"blood_vessel": 0, "glomerulus": 1, "unsure": 2}
    
    annotations_dict = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                annotations_dict[data['id']] = data['annotations']
    else:
        raise FileNotFoundError(f"Annotation file not found at {jsonl_path}")

    all_images = [f for f in os.listdir(raw_img_dir) if f.endswith('.tif')]
    meta_df = pd.read_csv(meta_csv_path) if os.path.exists(meta_csv_path) else None

    # Implement WSI-Aware Split to prevent data leakage
    if meta_df is not None and 'source_wsi' in meta_df.columns:
        # Filter data based on pipeline stage
        if stage == 1:
            # Teacher Model: Only use highly curated Dataset 1
            meta_df = meta_df[meta_df['dataset'] == 1]
            all_images = [img for img in all_images if os.path.splitext(img)[0] in set(meta_df['id'].astype(str))]
            
        wsi_unique = meta_df['source_wsi'].unique()
        train_wsis, val_wsis = train_test_split(wsi_unique, test_size=0.15, random_state=42)
        
        train_ids = set(meta_df[meta_df['source_wsi'].isin(train_wsis)]['id'].astype(str))
        val_ids = set(meta_df[meta_df['source_wsi'].isin(val_wsis)]['id'].astype(str))
        
        train_imgs = [img for img in all_images if os.path.splitext(img)[0] in train_ids]
        val_imgs = [img for img in all_images if os.path.splitext(img)[0] in val_ids]
    else:
        train_imgs, val_imgs = train_test_split(all_images, test_size=0.1, random_state=42)
    
    def process_split(img_list, split_name):
        img_dir = dirs[f'{split_name}_img']
        lbl_dir = dirs[f'{split_name}_lbl']
        
        for img_name in img_list:
            img_id = os.path.splitext(img_name)[0]
            src_img_path = os.path.join(raw_img_dir, img_name)
            dst_img_path = img_dir / img_name
            
            img = cv2.imread(src_img_path)
            if img is None:
                continue
            native_h, native_w = img.shape[:2] 
            
            if not dst_img_path.exists():
                shutil.copy(src_img_path, dst_img_path)
                
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
                        poly_np[:, 0] /= native_w
                        poly_np[:, 1] /= native_h
                        poly_np = np.clip(poly_np, 0.0, 1.0)
                        
                        flat_coords = poly_np.flatten().tolist()
                        coords_str = " ".join([f"{c:.6f}" for c in flat_coords])
                        lf.write(f"{class_id} {coords_str}\n")

    process_split(train_imgs, 'train')
    process_split(val_imgs, 'val')


def create_yaml_config(yolo_base_dir, yaml_path):
    config = {
        'path': os.path.abspath(yolo_base_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'blood_vessel', 1: 'glomerulus', 2: 'unsure'}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return os.path.abspath(yaml_path)


def main():
    parser = argparse.ArgumentParser(description="Train HuBMAP YOLOv11x-Seg")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=1, 
                        help="1: Teacher (DS1), 2: Pseudo-Label Generation, 3: Student (DS1 + DS2/DS3 Pseudo)")
    parser.add_argument("--weights", type=str, default="yolo11x-seg.pt")
    args = parser.parse_args()

    settings.update({'tensorboard': True})

    raw_img_dir = "data/train"
    # For stage 3, you would point this to a combined JSONL containing original DS1 + your generated pseudo-labels
    jsonl_path = "data/polygons.jsonl" if args.stage == 1 else "data/pseudo_polygons.jsonl"
    meta_csv_path = "data/tile_meta.csv"
    
    yolo_base_dir = f"datasets/hubmap_stage{args.stage}"
    yaml_path = f"{yolo_base_dir}/data.yaml"
    
    if not os.path.exists(yaml_path):
        prepare_yolo_dataset(raw_img_dir, jsonl_path, meta_csv_path, yolo_base_dir, stage=args.stage)
        create_yaml_config(yolo_base_dir, yaml_path)

    model = YOLO(args.weights)
    
    print(f"Starting Stage {args.stage} training for run: {args.run_name}")
    results = model.train(
        data=yaml_path,
        project=os.path.abspath("outputs"),
        name=args.run_name,
        device=0,
        imgsz=1024,
        batch=8,
        workers=8,
        epochs=100 if args.stage == 1 else 60,
        patience=20,
        optimizer="AdamW", 
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.05,
        cos_lr=True,
        warmup_epochs=3,
        hsv_h=0.02,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=45.0,
        translate=0.2,
        scale=0.3,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.25,
        mixup=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

if __name__ == "__main__":
    main()
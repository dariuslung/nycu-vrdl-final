import os
import glob
import json
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

import base64
import zlib
from pycocotools import _mask as coco_mask
import typing as t

def encode_binary_mask(mask: np.ndarray) -> str:
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask -- (Updated np.bool to bool for NumPy 1.20+ compatibility)
    if mask.dtype != bool:
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s" %
            mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" %
            mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    
    # Return as utf-8 string instead of bytes for clean CSV writing
    return base64_str.decode('utf-8')

def calculate_box_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def process_tta_predictions(model, image: np.ndarray, img_size: int = 1024):
    """
    Executes TTA using Original, Horizontal Flip, Vertical Flip, and Horizontal+Vertical Flip.
    """
    img_h, img_w = image.shape[:2]
    img_resized = cv2.resize(image, (img_size, img_size))
    
    # Generate the 4 augmented views
    views = {
        "orig": img_resized,
        "hflip": cv2.flip(img_resized, 1),
        "vflip": cv2.flip(img_resized, 0),
        "hvflip": cv2.flip(img_resized, -1)
    }
    
    all_boxes, all_scores, all_labels, all_masks = [], [], [], []
    
    for view_name, img_view in views.items():
        results = model.predict(img_view, iou=0.99, conf=0.01, retina_masks=True, verbose=False)[0]
        
        if results.boxes is None or results.masks is None:
            continue
            
        boxes = results.boxes.xyxyn.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy()
        masks = results.masks.data.cpu().numpy()  # Shape: (N, H, W)
        
        # Geometrically reverse the bounding boxes and masks to align with the original view
        if view_name == "hflip":
            boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]
            masks = np.flip(masks, axis=2)  # Axis 2 is width
        elif view_name == "vflip":
            boxes[:, [1, 3]] = 1.0 - boxes[:, [3, 1]]
            masks = np.flip(masks, axis=1)  # Axis 1 is height
        elif view_name == "hvflip":
            boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]
            boxes[:, [1, 3]] = 1.0 - boxes[:, [3, 1]]
            masks = np.flip(masks, axis=(1, 2))
            
        # Resize masks back to native WSI dimensions (512x512)
        masks_resized = np.array([
            cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_LINEAR) > 0.5 
            for m in masks
        ])
        
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        all_masks.append(masks_resized)
        
    return all_boxes, all_scores, all_labels, all_masks

def process_predictions(model, image: np.ndarray, img_size: int = 1024):
    img_h, img_w = image.shape[:2]
    img_resized = cv2.resize(image, (img_size, img_size))
    
    # Predict only on the original image
    results_orig = model.predict(img_resized, iou=0.99, conf=0.01, retina_masks=True, verbose=False)[0]
    
    all_boxes, all_scores, all_labels, all_masks = [], [], [], []
    
    if results_orig.boxes is not None and results_orig.masks is not None:
        all_boxes.append(results_orig.boxes.xyxyn.cpu().numpy()) 
        all_scores.append(results_orig.boxes.conf.cpu().numpy())
        all_labels.append(results_orig.boxes.cls.cpu().numpy())
        masks = results_orig.masks.data.cpu().numpy()
        
        # Resize masks back to original image dimensions
        masks_resized = np.array([cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_LINEAR) > 0.5 for m in masks])
        all_masks.append(masks_resized)

    return all_boxes, all_scores, all_labels, all_masks

def main():
    # Remember to change image_size
    weights_path = "/kaggle/input/datasets/dragozeroone/run6-5/best.pt"
    test_dir = "/kaggle/input/competitions/hubmap-hacking-the-human-vasculature/test"
    jsonl_path = "/kaggle/input/competitions/hubmap-hacking-the-human-vasculature/polygons.jsonl"
    output_csv = "submission.csv"
    
    iou_thr_wbf = 0.60
    skip_box_thr = 0.05
    min_pixel_area = 40  # Threshold for small object removal
    
    # --- UPGRADE: GROUND-TRUTH GLOMERULUS PRE-PARSING ---
    print("Pre-parsing human-annotated Glomerulus structures from metadata...")
    glom_annotations = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Select coordinates assigned strictly to glomeruli
                gloms = [anno for anno in data.get('annotations', []) if anno.get('type') == 'glomerulus']
                if gloms:
                    glom_annotations[data['id']] = gloms
    else:
        print(f"Warning: Metadata file not found at {jsonl_path}. Subtraction step will be bypassed.")

    model = YOLO(weights_path)
    test_images = glob.glob(os.path.join(test_dir, "*.tif"))
    submission_data = []
    
    for img_path in test_images:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        image = cv2.imread(img_path)
        img_h, img_w = image.shape[:2]
        
        boxes_list, scores_list, labels_list, masks_list = process_predictions(model, image, img_size=1408)
        # boxes_list, scores_list, labels_list, masks_list = process_tta_predictions(model, image, img_size=1408)
        
        # Optimization: Pre-filter arrays before WBF to prevent OOM scaling issues
        f_boxes_list, f_scores_list, f_labels_list, f_masks_list = [], [], [], []
        for b, s, l, m in zip(boxes_list, scores_list, labels_list, masks_list):
            keep_idx = s > 0.02
            f_boxes_list.append(b[keep_idx])
            f_scores_list.append(s[keep_idx])
            f_labels_list.append(l[keep_idx])
            f_masks_list.append(m[keep_idx])
            
        if sum(len(b) for b in f_boxes_list) == 0:
            submission_data.append({"id": img_id, "height": img_h, "width": img_w, "prediction_string": ""})
            continue
            
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            f_boxes_list, f_scores_list, f_labels_list, 
            # weights=[1, 1, 1, 1],
            # weights=[1, 1], 
            weights=[1], 
            iou_thr=iou_thr_wbf, 
            skip_box_thr=skip_box_thr
        )
        
        flat_boxes = np.concatenate(f_boxes_list)
        flat_masks = np.concatenate(f_masks_list)
        
        final_vessels = []
        
        for f_box, f_score, f_label in zip(fused_boxes, fused_scores, fused_labels):
            best_iou, best_mask_idx = 0, -1
            
            for idx, o_box in enumerate(flat_boxes):
                iou = calculate_box_iou(f_box, o_box)
                if iou > best_iou:
                    best_iou = iou
                    best_mask_idx = idx
            
            if best_mask_idx != -1:
                mask = flat_masks[best_mask_idx]
                if int(f_label) == 0:
                    final_vessels.append((mask, f_score))
                # Note: Class 1 is now 'unsure'. It is kept in training for regularization 
                # but omitted here during inference as it's not a target evaluation class.

        # --- UPGRADE: DETERMINISTIC GLOMERULUS MASK GENERATION ---
        glom_union_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        if img_id in glom_annotations:
            for anno in glom_annotations[img_id]:
                for poly in anno['coordinates']:
                    poly_np = np.array(poly, dtype=np.int32)
                    # Rasterize ground-truth polygons perfectly onto coordinate space
                    cv2.fillPoly(glom_union_mask, [poly_np], 1)
                    
        glom_union_mask = glom_union_mask.astype(bool)

        prediction_strings = []
        for vessel_mask, score in final_vessels:
            cleaned_vessel_mask = vessel_mask & ~glom_union_mask
            
            # Application of 7th Place methodology: Remove small noise fragments
            if np.sum(cleaned_vessel_mask) < min_pixel_area:
                continue
                
            # Application of YYama methodology: Dilation is strictly omitted to prevent Private LB degradation
            
            rle = encode_binary_mask(cleaned_vessel_mask.astype(bool))
            prediction_strings.append(f"0 {score:.4f} {rle}")
                
        submission_data.append({
            "id": img_id,
            "height": img_h,
            "width": img_w,
            "prediction_string": " ".join(prediction_strings)
        })

    df = pd.DataFrame(submission_data)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    main()
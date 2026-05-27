import os
import glob
import numpy as np
import pandas as pd
import cv2
import torch
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

# Ensure Kaggle offline compatibility: 
# You must attach the ensemble_boxes library as a dataset and add it to your path if not installed.
# sys.path.append('/kaggle/input/ensemble-boxes/ensemble_boxes-1.0.9')

def encode_binary_mask(mask: np.ndarray) -> str:
    """
    Converts a binary mask (numpy array) to Run-Length Encoding (RLE) 
    format required by the HuBMAP competition.
    """
    pixels = mask.flatten()
    # Pad with zeros to handle edges correctly
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def calculate_box_iou(box1, box2):
    """Calculates IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def process_tta_predictions(model, image: np.ndarray, img_size: int = 1024):
    """
    Runs TTA (Original + Horizontal Flip), extracts raw predictions bypassing strict NMS,
    and de-augments the results.
    """
    img_h, img_w = image.shape[:2]
    
    # Resize image to model input size
    img_resized = cv2.resize(image, (img_size, img_size))
    img_flipped = cv2.flip(img_resized, 1) # Horizontal flip
    
    # We set iou=0.99 and conf=0.01 to bypass YOLO's internal NMS as much as possible
    # retina_masks=True is critical to get high-resolution masks matching the image
    results_orig = model.predict(img_resized, iou=0.99, conf=0.01, retina_masks=True, verbose=False)[0]
    results_flip = model.predict(img_flipped, iou=0.99, conf=0.01, retina_masks=True, verbose=False)[0]
    
    all_boxes, all_scores, all_labels, all_masks = [], [], [], []
    
    # Process Original Image Predictions
    if results_orig.boxes is not None and results_orig.masks is not None:
        all_boxes.append(results_orig.boxes.xyxyn.cpu().numpy()) # Normalized boxes
        all_scores.append(results_orig.boxes.conf.cpu().numpy())
        all_labels.append(results_orig.boxes.cls.cpu().numpy())
        
        # Resize masks back to original image dimensions for accurate evaluation
        masks = results_orig.masks.data.cpu().numpy()
        masks_resized = np.array([cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_LINEAR) > 0.5 for m in masks])
        all_masks.append(masks_resized)

    # Process Flipped Image Predictions (De-augment)
    if results_flip.boxes is not None and results_flip.masks is not None:
        boxes_f = results_flip.boxes.xyxyn.cpu().numpy()
        # De-augment bounding boxes: flip x coordinates
        boxes_f[:, [0, 2]] = 1.0 - boxes_f[:, [2, 0]]
        
        all_boxes.append(boxes_f)
        all_scores.append(results_flip.boxes.conf.cpu().numpy())
        all_labels.append(results_flip.boxes.cls.cpu().numpy())
        
        masks_f = results_flip.masks.data.cpu().numpy()
        # De-augment masks: flip horizontally
        masks_f_deaug = np.flip(masks_f, axis=2)
        masks_f_resized = np.array([cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_LINEAR) > 0.5 for m in masks_f_deaug])
        all_masks.append(masks_f_resized)
        
    return all_boxes, all_scores, all_labels, all_masks

def main():
    # --- Configuration ---
    weights_path = "/kaggle/input/your-dataset/outputs/run_name/weights/best.pt"
    test_dir = "/kaggle/input/hubmap-hacking-the-human-vasculature/test"
    output_csv = "submission.csv"
    
    iou_thr_wbf = 0.60
    skip_box_thr = 0.05
    # ---------------------
    
    print("Loading YOLOv11-X model...")
    model = YOLO(weights_path)
    
    test_images = glob.glob(os.path.join(test_dir, "*.tif"))
    submission_data = []
    
    for img_path in test_images:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        image = cv2.imread(img_path)
        img_h, img_w = image.shape[:2]
        
        # 1. Run TTA and extract dense predictions
        boxes_list, scores_list, labels_list, masks_list = process_tta_predictions(model, image)
        
        if not boxes_list:
            submission_data.append({"id": img_id, "height": img_h, "width": img_w, "prediction_string": ""})
            continue
            
        # 2. Apply Weighted Boxes Fusion
        # WBF requires coordinates to be normalized between [0, 1]
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, 
            weights=[1, 1], # Equal weights for Original and H-Flip
            iou_thr=iou_thr_wbf, 
            skip_box_thr=skip_box_thr
        )
        
        # Flatten original data for mask mapping
        flat_boxes = np.concatenate(boxes_list)
        flat_masks = np.concatenate(masks_list)
        
        final_vessels = []
        final_glomeruli = []
        
        # 3. Map Fused Boxes back to Masks
        for f_box, f_score, f_label in zip(fused_boxes, fused_scores, fused_labels):
            best_iou = 0
            best_mask_idx = -1
            
            # Find the original box that best matches the fused box
            for idx, o_box in enumerate(flat_boxes):
                iou = calculate_box_iou(f_box, o_box)
                if iou > best_iou:
                    best_iou = iou
                    best_mask_idx = idx
            
            if best_mask_idx != -1:
                mask = flat_masks[best_mask_idx]
                if int(f_label) == 0:    # blood_vessel
                    final_vessels.append((mask, f_score))
                elif int(f_label) == 1:  # glomerulus
                    final_glomeruli.append(mask)

        # 4. Glomerulus Subtraction (FP Reduction Trick)
        # Create a unified mask of all glomeruli to subtract from vessels
        glom_union_mask = np.zeros((img_h, img_w), dtype=bool)
        if final_glomeruli:
            glom_union_mask = np.any(final_glomeruli, axis=0)

        # 5. Format Prediction String
        prediction_strings = []
        for vessel_mask, score in final_vessels:
            # Subtract glomerulus structures from the blood vessel
            cleaned_vessel_mask = vessel_mask & ~glom_union_mask
            
            # Only encode if the mask still has area after subtraction
            if np.any(cleaned_vessel_mask):
                rle = encode_binary_mask(cleaned_vessel_mask)
                # Kaggle format: class_id confidence rle
                prediction_strings.append(f"0 {score:.4f} {rle}")
                
        prediction_string = " ".join(prediction_strings)
        submission_data.append({
            "id": img_id,
            "height": img_h,
            "width": img_w,
            "prediction_string": prediction_string
        })
        
        print(f"Processed {img_id}: found {len(final_vessels)} vessels after WBF.")

    # 6. Save Submission
    df = pd.DataFrame(submission_data)
    df.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")

if __name__ == "__main__":
    main()
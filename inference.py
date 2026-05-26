import base64
import os
import typing as t
import zlib

import cv2
import numpy as np
import pandas as pd
from pycocotools import _mask as coco_mask
from ultralytics import YOLO


def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""
    if mask.dtype != bool and mask.dtype != np.bool_:
        raise ValueError(f"encode_binary_mask expects a binary mask, received dtype == {mask.dtype}")

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(f"encode_binary_mask expects a 2d mask, received shape == {mask.shape}")

    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    
    return base64_str.decode('utf-8')


def main():
    test_dir = "/kaggle/input/competitions/hubmap-hacking-the-human-vasculature/test"
    model_path = "/kaggle/input/datasets/dragozeroone/run2-model/best.pt" 
    
    print("Loading model...")
    model = YOLO(model_path)
    submission_data = []
    
    # Keep confidence low to maximize recall for the AP metric
    CONFIDENCE_THRESHOLD = 0.01 
    
    print("Starting inference...")
    for img_name in os.listdir(test_dir):
        if not img_name.endswith('.tif'):
            continue
            
        img_id = img_name.split('.')[0]
        img_path = os.path.join(test_dir, img_name)
        
        # FIX: Pass the file path directly to YOLO. 
        # This allows Ultralytics to natively handle the BGR->RGB conversion safely.
        results = model(img_path, imgsz=512, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        
        height, width = results.orig_shape
        
        # 1. Build the Glomerulus Exclusion Mask
        glom_mask = np.zeros((height, width), dtype=bool)
        if results.masks is not None:
            for i, poly in enumerate(results.masks.xy):
                cls_id = int(results.boxes.cls[i].item())
                if cls_id == 1:  # glomerulus
                    if len(poly) > 0:
                        # FIX: Use np.round to prevent coordinate shift from truncation
                        poly_pts = np.round(poly).astype(np.int32)
                        temp_glom = np.zeros((height, width), dtype=np.uint8)
                        cv2.fillPoly(temp_glom, [poly_pts], 1)
                        glom_mask = glom_mask | (temp_glom == 1)

        # 2. Process Blood Vessels
        pred_strings = []
        if results.masks is not None:
            for i, poly in enumerate(results.masks.xy):
                cls_id = int(results.boxes.cls[i].item())
                conf = float(results.boxes.conf[i].item())
                
                if cls_id == 0:  # blood_vessel
                    if len(poly) > 0:
                        poly_pts = np.round(poly).astype(np.int32)
                        
                        vessel_mask = np.zeros((height, width), dtype=np.uint8)
                        cv2.fillPoly(vessel_mask, [poly_pts], 1)
                        vessel_mask_bool = vessel_mask.astype(bool)
                        
                        # Apply exclusion zone
                        vessel_mask_bool = vessel_mask_bool & (~glom_mask)
                        
                        # Only encode and submit if the vessel still exists after masking
                        if vessel_mask_bool.any():
                            encoded_str = encode_binary_mask(vessel_mask_bool)
                            pred_strings.append(f"0 {conf:.4f} {encoded_str}")
        
        prediction_string = " ".join(pred_strings)
        
        submission_data.append({
            "id": img_id,
            "height": height,
            "width": width,
            "prediction_string": prediction_string
        })
        
    df = pd.DataFrame(submission_data)
    df.to_csv("submission.csv", index=False)
    print(f"Inference complete. Processed {len(df)} images. submission.csv generated.")


if __name__ == "__main__":
    main()
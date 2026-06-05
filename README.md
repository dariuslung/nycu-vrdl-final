# NYCU Visual Recognition using Deep Learning - Spring 2026 - Final

- Student ID: 314561005
- Name: 龍偉亮
- Competition: [HuBMAP - Hacking the Human Vasculature](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature)

## Introduction

This repository contains a YOLO-based segmentation pipeline designed for the HuBMAP - Hacking the Human Vasculature competition. The objective is to identify and segment microvascular structures (`blood_vessel`) from histological images while explicitly excluding `glomerulus` structures and ignoring `unsure` regions.

The pipeline features a two-stage training approach with a Whole Slide Image (WSI)-aware spatial split. The inference module utilizes Weighted Boxes Fusion (WBF), morphological opening to filter noise, and deterministic ground-truth subtraction to generate properly encoded RLE submission strings.

## Environment Setup

The required dependencies are managed via Conda. The environment relies on Python 3.11, PyTorch compiled for CUDA 12.4, and the Ultralytics YOLO framework.

**Installation Steps:**

1. Ensure Conda (or Mamba) is installed on your system.
2. Create the environment from the provided YAML configuration:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate dalius_ml
```

## Usage

### 1. Data Preparation

Before running the training script, ensure your raw dataset is located in a `data/` directory at the root of the project. It requires the following structure:

* `data/train/` (Directory containing the `.tif` images)
* `data/polygons.jsonl` (Annotation data)
* `data/tile_meta.csv` (Metadata required for the spatial split logic)

### 2. Training

The training script (`train.py`) automatically constructs the YOLO-compatible directory structure, parses the polygons, and executes the training loop.

**Basic Command:**

```bash
python train.py --run_name my_training_run --stage 1

```

**Arguments:**

* `--run_name`: Name of the training run to track in TensorBoard/outputs (Required).
* `--stage`: Set to `1` (Trains on Dataset 1 & 2) or `2` (Trains on Dataset 1 only). Default is `1`.
* `--weights`: Path to initial weights. Default is `yolo11x-seg.pt`.

**Configuring the Validation Split:**
To prevent data leakage, the pipeline utilizes a strict spatial split on WSI 1 based on the tile coordinates (`i`). You can dictate which half of WSI 1 is used for the validation set by modifying `train.py`.

Locate the following block in the `prepare_yolo_dataset` function:

```python
median_i = wsi_1_df['i'].median()
# val_ids = set(wsi_1_df[wsi_1_df['i'] < median_i]['id'].astype(str))
val_ids = set(wsi_1_df[wsi_1_df['i'] >= median_i]['id'].astype(str))

```

* **Right Half (Default):** The active line `...['i'] >= median_i...` assigns the right half of the slide to the validation set.
* **Left Half:** To validate on the left half of the slide instead, uncomment the line `val_ids = set(wsi_1_df[wsi_1_df['i'] < median_i]['id'].astype(str))` and comment out the `>=` line below it.

### 3. Inference

The inference script (`inference.py`) evaluates test images, aggregates predictions using an ensemble of models or Test Time Augmentation (TTA), applies morphological cleaning (erosion/dilation), and outputs the required `submission.csv`.

**Execution:**

```bash
python inference.py

```

**Important Note:** Before executing inference, you must manually update the hardcoded file paths at the top of the `main()` function in `inference.py` to point to your trained weights and test directories:

* `weights_path`
* `weights2_path`
* `test_dir`
* `jsonl_path`

## Performance Snapshot

<img width="1084" height="121" alt="image" src="https://github.com/user-attachments/assets/9ca5299b-097a-4d61-98bb-813b486d3b10" />

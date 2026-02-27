# Car Damage / Object Detection Pipeline (BDD100K) – Faster R-CNN Baseline + Experiments

## Overview
This project implements an end-to-end object detection pipeline based on Faster R-CNN (ResNet50-FPN) trained on the BDD100K dataset. The primary goal is to build a strong, research-oriented baseline while systematically experimenting with training stability, class imbalance handling, sampling strategies, and evaluation metrics.

All code is contained within a single Kaggle notebook for reproducibility and ease of experimentation.

---

## Key Objectives
- Train a robust Faster R-CNN detector on BDD100K
- Handle class imbalance effectively
- Improve training stability via warmup scheduling
- Reduce redundant bounding boxes
- Introduce proper evaluation metrics (mAP instead of only validation loss)
- Create a modular pipeline suitable for future extensions (e.g., damage classification after detection)

---

## Dataset
### BDD10K (A subset of BDD100k due to resource limitations)
- Dataset Size (Train): ~7000 images (custom filtered subset)
- Task: Multi-class object detection
- Classes include: cars, pedestrians, trucks, buses, etc.
- Original image resolution is preserved (no resizing)

### Preprocessing
- Simple Albumentations-based pipeline
- Normalization only:
  - Mean: (0.485, 0.456, 0.406)
  - Std: (0.229, 0.224, 0.225)
- ToTensorV2 conversion
- No explicit resizing (model handles multi-scale internally via FPN)

---

## Model Architecture
### Base Model
- Faster R-CNN with ResNet50 + FPN backbone
- Pretrained weights: DEFAULT (ImageNet + COCO pretraining)

---

## Handling Class Imbalance (Core Experiment)
### Repeat Factor Sampling (RFS)
Implemented custom Repeat Factor Sampler inspired by LVIS methodology.

### Motivation
BDD10K exhibits heavy class imbalance, in which the number of car occurences are hundreds of times more frequent than trains and trucks. Examples are:
- Frequent: cars, pedestrians 
- Rare: buses, trucks, etc.

### Implementation Details
- Compute per-image repeat factors based on class frequency
- Threshold tuning (tested: 0.08 → 0.05)
- Custom sampler replaces default shuffle

### Observed Stats
- Original Dataset: 7000
- Virtual Dataset (RFS): Around 7500
- Mean Repeat Factor: ~1.05
- Max Repeat Factor: ~3

### Interpretation:
- Mild oversampling of rare classes
- Stable and expected behavior (not over-aggressive)

---

## Evaluation Strategy
Validation Loss is most often not enough for tracking how well a model is doing and given that Object detection loss does not directly correlate with detection quality. Therefore, the primary metric I decided to use is: 

- mAP (mean Average Precision)

### Secondary Monitoring:
- Validation loss (for optimization diagnostics)

  
Evaluated periodically (for every few epochs)
More aligned with real detection performance
Trade-off: Slightly slower but significantly more informative

---

## Anchor & Resolution Considerations
- No image resizing applied
- Original BDD resolution retained
- Default Faster R-CNN anchors are unstable for BDD10k objects. My assumption is that the base anchors were primarily optimized for ImageNET, in which generally has quite large objects. BDD10k has many small objects, so I decided to add additional smaller anchors while removing the largest one. However, this is still mainly experimental for now.

---

## Current Limitations
Dataset size relatively small (7k images)
No advanced augmentations (e.g., Mosaic, MixUp)
Anchor size is primarily determined using assumptions and not true Statistics.

## Future Directions
- Anchor size auto-tuning based on dataset statistics
- Advanced augmentations (CutMix, Mosaic)
- Focal Loss integration for imbalance
- Soft-NMS / DIoU-NMS
- Larger training subset of BDD100K

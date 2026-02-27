# 🚗 Car Damage Severity Classification (Ordinal-Aware Deep Learning)

## Project Overview
This project focuses on building an ordinal-aware image classification pipeline to predict **car damage severity** (Minor, Moderate, Severe) using a deep learning model trained on the Car Damage Severity Dataset (Kaggle).

Unlike standard classification projects, this work treats damage severity BOTH as a **classification problem** AND as an **ordinal problem**, and introduces custom evaluation metrics and loss modifications to better reflect real-world risk priorities (e.g., misclassifying Severe as Minor is worse than Moderate → Severe).

All experiments, EDA, training, and evaluation are implemented in a **single Kaggle Notebook** for full reproducibility.

---

## Problem Framing
### Task Type:
Ordinal Image Classification (NOT just standard multi-class classification)

### Classes (Encoded):
| Class Name | Label |
|-----------|--------|
| Minor     | 0      |
| Moderate  | 1      |
| Severe    | 2      |

Key Insight:
- These classes have **ordered semantic meaning**
- Traditional CrossEntropy treats all mistakes equally
- But in real-world insurance / automotive AI:
  - Predicting Severe as Minor is a critical error
  - Predicting Severe as Moderate is less severe

This motivated the use of **ordinal-aware modeling and evaluation**.

---

## Dataset

### Dataset Statistics
- Minor: 452 images  
- Moderate: 463 images  
- Severe: 468 images  


## Exploratory Data Analysis (EDA)
Performed:
- Class distribution analysis
- Random image visualization per class
- Dataset balance verification
- Label encoding inspection

## Model Pipeline
### Training Workflow:
1. Load images from class folders
2. Encode class names → numeric ordinal labels
3. Apply preprocessing & normalization
4. Train CNN-based classifier
5. Evaluate using BOTH classification and ordinal metrics


## Unique & Advanced Techniques Implemented

### Ordinal-Aware Evaluation (Beyond Accuracy)
Instead of relying only on accuracy, the project tracks:

- Validation Accuracy
- **Ordinal MAE (Mean Absolute Error)**
- Per-Class Recall (Minor / Moderate / Severe)

Why MAE?
Because:
- Pred = Severe (2), GT = Minor (0) → Error = 2 (Very bad)
- Pred = Moderate (1), GT = Severe (2) → Error = 1 (Less severe)

This better reflects real-world model usefulness.

---

### Distance-Weighted Cross Entropy Loss (Custom Loss)
Implemented a hybrid loss:
$ Total Loss = CrossEntropy + λ × Ordinal Distance Penalty $

Benefits:
- Penalizes large ordinal mistakes more heavily
- Encourages the model to learn severity hierarchy
- Improves Moderate vs Severe discrimination (key weakness observed)

  
### Per-Class Recall Monitoring (Critical Insight)
Tracked recall separately for:
- Minor Recall
- Moderate Recall
- Severe Recall

**Key Finding:**
> The model initially over-predicted Severe and struggled to distinguish Moderate vs Severe, which makes sense considering the line between an object being moderately damaged and severely damaged is quite blurry.

This level of diagnostic analysis demonstrates **professional ML evaluation practice**.


## Potential Future Improvements
- Ordinal Regression (CORAL / Cumulative Link Models)
- Label smoothing for noisy severity labels
- Hard example mining for Moderate vs Severe cases
- Ensemble with object-detection + crop pipeline (Faster R-CNN + classifier)
- Grad-CAM explainability for model interpretability

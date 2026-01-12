# K-Fold Cross-Validation Guide

## Overview

This guide explains how to use stratified k-fold cross-validation with the brain tumor detection dataset. The implementation preserves class distribution and maintains patient-level integrity (all images from the same patient stay together).

## Correct K-Fold CV Workflow

The proper way to use k-fold cross-validation:

1. **First**: Split data into train+val (80%) and test (20%) using `split_data_preserve_distribution()`
2. **Second**: Apply k-fold CV **only on train+val** to get robust validation metrics
3. **Finally**: Evaluate the best model on the **held-out test set** (never touched during CV)

```
Full Dataset (233 patients)
├── Train+Val (80%, ~186 patients) ──► K-Fold CV here
│   ├── Fold 0: Train (90%) / Val (10%)
│   ├── Fold 1: Train (90%) / Val (10%)
│   └── ... (k folds)
└── Test (20%, ~47 patients) ──► Final evaluation only
```

## Function: `split_data_kfold_cv`

Located in `preprocessing/split.py`, this function creates k folds for cross-validation:

- **Input**: Train+val data (output from `split_data_preserve_distribution()`)
- **Output**: k folds, each with train/val splits
- **Stratified**: Preserves class distribution across all folds
- **Patient-level integrity**: All images from the same patient stay together
- **Reproducible**: Uses fixed random seed (default 42)

## How It Works

### 1. Stratified Folding
- Groups patients by tumor type (Meningioma, Glioma, Pituitary Tumor)
- For each tumor type, splits patients into k folds
- Ensures each fold has approximately the same class distribution

### 2. Train/Val Splits per Fold
For each fold i:
- **Val set**: Fold i from all tumor types (~10% for k=10)
- **Train set**: Remaining k-1 folds (~90% for k=10)

### 3. Output Structure

```
output_path/
├── fold_0/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
├── fold_1/
│   └── ...
└── fold_9/
    └── ...
```

## Usage

### Step 1: Create Initial Split (Train+Val and Test)

First, split the full dataset to hold out a test set:

```python
from preprocessing.split import split_data_preserve_distribution

# Split full dataset: 80% train+val, 20% test
split_data_preserve_distribution(
    image_path='../output/images',
    label_path='../output/labels',
    output_path='../output/btf_stratified',
    train_ratio=0.8,  # This becomes our "train+val" for k-fold
    val_ratio=0.0     # No separate val yet (k-fold will create these)
)
# Note: test_ratio = 1 - 0.8 - 0.0 = 0.2
```

Or manually combine train+val from an existing split into a single folder.

### Step 2: Apply K-Fold CV on Train+Val

```python
from preprocessing.split import split_data_kfold_cv

# Create 10-fold CV from the train+val portion
split_data_kfold_cv(
    trainval_image_path='../output/btf_stratified/images',      # or path to combined train+val
    trainval_label_path='../output/btf_stratified/labels',      # or path to combined train+val labels
    output_path='../output/kfold_cv',
    k=10,
    seed=42
)
```

### Parameters

- `trainval_image_path`: Path to train+val images (patient ID subfolders)
- `trainval_label_path`: Path to train+val labels (patient ID subfolders)
- `output_path`: Path where fold directories will be created
- `k`: Number of folds (default: 10)
- `seed`: Random seed for reproducibility (default: 42)

### Custom K-Fold (e.g., 5-fold)

```python
split_data_kfold_cv(
    trainval_image_path='../output/btf_stratified/images',
    trainval_label_path='../output/btf_stratified/labels',
    output_path='../output/kfold_cv_5fold',
    k=5,
    seed=42
)
```

## Training with K-Fold CV

### Example Training Loop

```python
import os
import numpy as np
from ultralytics import YOLO

val_maps = []

# Train on each fold
for fold_idx in range(10):
    fold_path = f'../output/kfold_cv/fold_{fold_idx}'
    
    # Train model
    model = YOLO('yolo11l.pt')
    model.train(
        data=f'{fold_path}/train',
        epochs=100,
        imgsz=640,
        project='runs/kfold_cv',
        name=f'fold_{fold_idx}',
        val=f'{fold_path}/val'
    )
    
    # Evaluate on this fold's validation set
    results = model.val(data=f'{fold_path}/val')
    val_maps.append(results.box.map50)
    print(f'Fold {fold_idx} - Val mAP50: {results.box.map50}')

# Report cross-validation results
print(f'\nCross-Validation Results:')
print(f'Mean Val mAP50: {np.mean(val_maps):.4f} ± {np.std(val_maps):.4f}')
```

### Final Evaluation on Held-Out Test Set

After k-fold CV, evaluate your best model on the test set:

```python
# Load best model (e.g., from best fold or retrained on all train+val)
model = YOLO('runs/kfold_cv/fold_best/weights/best.pt')

# Evaluate on held-out test set (ONLY ONCE, at the very end)
test_results = model.val(data='../output/btf_stratified/test')
print(f'Final Test mAP50: {test_results.box.map50}')
```

## Advantages of K-Fold CV

1. **Robust Validation**: Each patient in train+val appears in validation exactly once
2. **Reduced Variance**: Averaging across k folds reduces impact of a particular split
3. **Maximizes Data**: All train+val data used for both training and validation
4. **Stratified**: Maintains class distribution across all folds
5. **Unbiased Test**: Held-out test set never influences model selection

## Comparison with Single Split

| Aspect | Single Split | K-Fold CV |
|--------|--------------|-----------|
| Validation set size | 10% of data | 10% per fold (all train+val validated) |
| Training iterations | 1 | k (e.g., 10) |
| Variance in val results | Higher | Lower (averaged) |
| Computational cost | Lower | k times higher |
| Test set | 20% held out | 20% held out (same) |
| Best for | Quick experiments | Robust validation, publication |

## Best Practices

1. **Hold out test set first**: Never include test data in k-fold CV
2. **Keep seed fixed**: Use the same seed (42) for reproducibility
3. **Monitor class distribution**: Check printed output for each fold
4. **Save results per fold**: Track metrics separately to identify issues
5. **Report mean ± std**: For robust performance estimates
6. **Final test evaluation**: Only evaluate on test set once, after all tuning

## Notes

- The function automatically handles patients with no tumors (skips with warning)
- Each fold maintains approximately the same class distribution
- Patient-level splitting ensures no data leakage
- The held-out test set should only be used for final evaluation

## Related Functions

- `split_data()`: Simple ratio-based splitting (70/10/20)
- `split_data_preserve_distribution()`: Single split with preserved class distribution (use this first!)
- `split_data_real_world_distribution()`: Greedy algorithm with specific target counts


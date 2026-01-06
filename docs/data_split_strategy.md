# Data Split Strategy

## Overview
This document describes the data splitting methodology used for the brain tumor detection project.

## Split Method: Patient-Level Splitting

**Important:** The dataset is split **by patient ID**, not by individual images. This means all images from the same patient remain together in the same split (train/val/test).

### Why Patient-Level Splitting?
This approach is crucial for medical imaging datasets to prevent **data leakage**:
- Images from the same patient often share similar characteristics (imaging conditions, anatomy, etc.)
- If images from the same patient appear in both training and test sets, the model may achieve artificially high performance by recognizing patient-specific features rather than learning to detect tumors
- Patient-level splitting ensures the model is evaluated on truly unseen patients, providing a more realistic assessment of generalization performance

## Split Configuration

### Ratios
- **Train:** 70% of patients
- **Validation:** 10% of patients
- **Test:** 20% of patients

### Reproducibility
- **Random seed:** 42 (fixed)
- **Method:** Patients are shuffled using `np.random.seed(42)` before splitting
- This ensures the exact same split can be reproduced in future runs

## Implementation Details

The splitting logic is implemented in `preprocessing/split.py`:

```python
split_data(
    image_path='path/to/images',
    label_path='path/to/labels',
    output_path='path/to/output',
    train_ratio=0.7,
    val_ratio=0.1
)
```

### Process
1. Identify all patient ID folders
2. Sort patient IDs alphabetically
3. Shuffle patient IDs using fixed seed (42)
4. Calculate split points based on patient count
5. Assign patients to train/val/test splits
6. Copy all images and labels for each patient to their respective split folders

## Actual Split Results

Based on the current dataset split (executed on 2025-11-13):

| Split      | Patients | Images | Percentage (Patients) |
|------------|----------|--------|-----------------------|
| Train      | 163      | 2,130  | 69.96%               |
| Validation | 23       | 309    | 9.87%                |
| Test       | 47       | 625    | 20.17%               |
| **Total**  | **233**  | **3,064** | **100%**            |

### Observations
- Average images per patient: ~13.1 images
- The distribution of images across splits may not exactly match the patient ratio due to varying numbers of images per patient
- Patient-level splitting ensures no patient appears in multiple splits

## Best Practices Followed

- **Patient-level splitting** prevents data leakage  
- **Fixed random seed** ensures reproducibility  
- **Validation set** allows hyperparameter tuning without touching test set  
- **Separate test set** provides unbiased final evaluation  
- **Documentation** of split methodology and results  

## Notes for Future Work

- If retraining or experimenting with different splits, always use the same seed (42) to maintain consistency
- If you need a different split ratio, modify the `train_ratio` and `val_ratio` parameters in the function call
- The test ratio is automatically calculated as: `1 - train_ratio - val_ratio`
- Always verify that no patient IDs overlap between splits after any modifications

## Related Files

- Implementation: `preprocessing/split.py`
- Output directory: `output/` (or specified output path)


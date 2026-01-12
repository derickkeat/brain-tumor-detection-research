# Data Split Strategy

## Overview
This document describes the data splitting methodology used for the brain tumor detection project.

## Dataset Distribution

### Overall Tumor Type Distribution

The dataset contains **3,064 tumors** across **3,064 label files** (one tumor per image):

| Tumor Type           | Class ID | Count     | Percentage |
|----------------------|----------|-----------|------------|
| Meningioma           | 0        | 708       |  23.11%    |
| Glioma               | 1        | 1,426     |  46.54%    |
| Pituitary Tumor      | 2        | 930       |  30.35%    |
| **Total**            |          | **3,064** |  **100%**  |

### Balance Analysis

- **Minimum tumor count:** 708 (Meningioma)
- **Maximum tumor count:** 1,426 (Glioma)
- **Average tumor count:** 1,021.33
- **Imbalance ratio (max/min):** 2.01

⚠️ **Note:** Significant class imbalance detected. Glioma is the most common tumor type (46.54%), while Meningioma is the least common (23.11%). Consider using class weights or data augmentation during training to address this imbalance.

*Distribution calculated using `utils/inspect_tumor_distribution.py`*

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

## Real-World Distribution Split Method: Class-Balanced Patient Splitting

An alternative splitting method that ensures balanced class distribution across splits using a greedy algorithm.

### Overview

The real-world distribution split method (`split_data_real_world_distribution`) distributes patients based on their primary tumor type to achieve target counts for each class in test and validation sets.

### Target Distribution

**Test Set Targets:**
- Meningioma: 23 patients
- Glioma: 14 patients
- Pituitary Tumor: 10 patients

**Validation Set Targets:**
- Meningioma: 11 patients
- Glioma: 7 patients
- Pituitary Tumor: 5 patients

**Training Set:**
- All remaining patients (not assigned to test or val)

### Algorithm

1. **Shuffle patients** using fixed seed (42) for reproducibility
2. **Determine primary tumor type** for each patient:
   - Count tumors of each class (0=meningioma, 1=glioma, 2=pituitary_tumor) across all patient's label files
   - Primary type is the class with the most tumors
3. **Greedy assignment** (process patients in shuffled order):
   - Try to add patient to **test set** first
     - If adding would exceed target for patient's tumor type → skip test
   - Try to add patient to **val set** next
     - If adding would exceed target for patient's tumor type → skip val
   - Add patient to **train set** (all remaining patients)

### Implementation

```python
split_data_real_world_distribution(
    image_path='path/to/images',
    label_path='path/to/labels',
    output_path='path/to/output'
)
```

### Key Features

- **Class-balanced splits**: Ensures specific number of patients per tumor type in test/val
- **Patient-level splitting**: All images from same patient stay together
- **Deterministic**: Uses fixed seed (42) for reproducibility
- **Greedy approach**: Processes patients sequentially, filling test first, then val, then train

### Use Cases

Use the real-world distribution split method when:
- You need balanced class distribution across splits
- You have specific requirements for test/val set composition
- You want to ensure sufficient representation of each tumor type in evaluation sets

Use the ratio-based split method when:
- You want a simple proportional split
- Class balance is less critical
- You prefer a more random distribution

## Notes for Future Work

- If retraining or experimenting with different splits, always use the same seed (42) to maintain consistency
- If you need a different split ratio, modify the `train_ratio` and `val_ratio` parameters in the function call
- The test ratio is automatically calculated as: `1 - train_ratio - val_ratio`
- Always verify that no patient IDs overlap between splits after any modifications
- For real-world distribution split, adjust target counts in `split_data_real_world_distribution()` function if different class distribution is needed

## Related Files

- Implementation: `preprocessing/split.py`
  - `split_data()`: Ratio-based splitting
  - `split_data_real_world_distribution()`: Class-balanced greedy splitting
- Output directory: `output/` (or specified output path)


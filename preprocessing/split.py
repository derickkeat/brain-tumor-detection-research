import numpy as np

import os
import glob
import shutil

def split_data(image_path, label_path, output_path, train_ratio=0.7, val_ratio=0.1):
    """
    Split the data into train, val, and test sets by patient ID.
    All images from the same patient stay together in the same split.
    
    Args:
        image_path: Path to folder containing patient ID subfolders with images
        label_path: Path to folder containing patient ID subfolders with labels
        output_path: Path where split folders will be created
        train_ratio: Proportion of patients for training (default 0.7)
        val_ratio: Proportion of patients for validation (default 0.1)
        test_ratio: Remainder goes to test (default 0.2)
    """
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)
    
    # Get all patient ID folders
    patient_folders = [d for d in os.listdir(image_path) 
                      if os.path.isdir(os.path.join(image_path, d))]
    patient_ids = sorted(patient_folders)
    
    # Shuffle patient IDs with a fixed seed for reproducibility
    np.random.seed(42)
    shuffled_patients = np.random.permutation(patient_ids)
    
    # Calculate split points based on number of patients
    n_patients = len(patient_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    # Split patient IDs
    train_patients = shuffled_patients[:n_train]
    val_patients = shuffled_patients[n_train:n_train + n_val]
    test_patients = shuffled_patients[n_train + n_val:]
    
    # Function to copy all files from patients to split folders
    def copy_split(patient_list, split_name):
        total_images = 0
        for patient_id in patient_list:
            # Get all images for this patient
            patient_image_folder = os.path.join(image_path, patient_id)
            image_files = glob.glob(os.path.join(patient_image_folder, '*.jpg'))
            
            for image_file in image_files:
                base_name = os.path.splitext(os.path.basename(image_file))[0]
                
                # Copy image
                dst_image = os.path.join(output_path, split_name, 'images', f'{base_name}.jpg')
                shutil.copy2(image_file, dst_image)
                
                # Copy corresponding label
                src_label = os.path.join(label_path, patient_id, f'{base_name}.txt')
                dst_label = os.path.join(output_path, split_name, 'labels', f'{base_name}.txt')
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
                else:
                    print(f'Warning: Label file not found for patient {patient_id}, image {base_name}')
                
                total_images += 1
        
        return total_images
    
    # Copy files to respective splits
    train_count = copy_split(train_patients, 'train')
    val_count = copy_split(val_patients, 'val')
    test_count = copy_split(test_patients, 'test')
    
    print(f'Split complete (by patient ID):')
    print(f'  Train: {len(train_patients)} patients ({train_count} images)')
    print(f'  Val: {len(val_patients)} patients ({val_count} images)')
    print(f'  Test: {len(test_patients)} patients ({test_count} images)')
    print(f'  Total: {n_patients} patients ({train_count + val_count + test_count} images)')


def split_data_real_world_distribution(image_path, label_path, output_path):
    """
    Split the data into train, val, and test sets using a greedy algorithm.
    The algorithm ensures balanced class distribution across splits by counting patients:
    - Test set targets: 23 meningioma patients, 14 glioma patients, 10 pituitary_tumor patients
    - Val set targets: 11 meningioma patients, 7 glioma patients, 5 pituitary_tumor patients
    - Remaining patients go to train set
    
    Each patient's tumor type is determined by the most common tumor class in their labels.
    All images from the same patient stay together in the same split.
    
    Args:
        image_path: Path to folder containing patient ID subfolders with images
        label_path: Path to folder containing patient ID subfolders with labels
        output_path: Path where split folders will be created
    """
    
    # Class mapping: 0=meningioma, 1=glioma, 2=pituitary_tumor
    CLASS_NAMES = {0: 'meningioma', 1: 'glioma', 2: 'pituitary_tumor'}
    
    # Target counts for each split
    test_targets = {0: 23, 1: 14, 2: 10}  # meningioma, glioma, pituitary_tumor
    val_targets = {0: 11, 1: 7, 2: 5}
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)
    
    # Get all patient ID folders
    patient_folders = [d for d in os.listdir(image_path) 
                      if os.path.isdir(os.path.join(image_path, d))]
    patient_ids = sorted(patient_folders)
    
    # Shuffle patient IDs with a fixed seed for reproducibility
    np.random.seed(42)
    shuffled_patients = np.random.permutation(patient_ids)
    
    # Function to determine primary tumor type for a patient
    def get_patient_tumor_type(patient_id):
        """
        Determine the primary tumor type for a patient.
        Returns the class ID (0, 1, or 2) with the most tumors, or None if no tumors found.
        """
        counts = {0: 0, 1: 0, 2: 0}  # meningioma, glioma, pituitary_tumor
        
        patient_label_folder = os.path.join(label_path, patient_id)
        if not os.path.isdir(patient_label_folder):
            return None
        
        label_files = glob.glob(os.path.join(patient_label_folder, '*.txt'))
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 1:
                                class_id = int(parts[0])
                                if class_id in [0, 1, 2]:
                                    counts[class_id] += 1
            except Exception as e:
                print(f'Warning: Error reading label file {label_file}: {e}')
        
        # Return the class with the most tumors, or None if no tumors found
        max_count = max(counts.values())
        if max_count == 0:
            return None
        
        # If there's a tie, return the first one encountered (0, then 1, then 2)
        for class_id in [0, 1, 2]:
            if counts[class_id] == max_count:
                return class_id
        
        return None
    
    # Initialize split lists and patient counts by tumor type
    train_patients = []
    val_patients = []
    test_patients = []
    
    # Count patients (not tumors) by tumor type in each split
    test_patient_counts = {0: 0, 1: 0, 2: 0}  # meningioma, glioma, pituitary_tumor
    val_patient_counts = {0: 0, 1: 0, 2: 0}
    
    # Greedy algorithm: process each patient in shuffled order
    for patient_id in shuffled_patients:
        patient_tumor_type = get_patient_tumor_type(patient_id)
        
        # Skip patients with no tumors
        if patient_tumor_type is None:
            train_patients.append(patient_id)
            continue
        
        # Check if adding this patient to test would exceed the target for their tumor type
        if test_patient_counts[patient_tumor_type] < test_targets[patient_tumor_type]:
            # Add to test set
            test_patients.append(patient_id)
            test_patient_counts[patient_tumor_type] += 1
        elif val_patient_counts[patient_tumor_type] < val_targets[patient_tumor_type]:
            # Add to val set
            val_patients.append(patient_id)
            val_patient_counts[patient_tumor_type] += 1
        else:
            # Add to train set
            train_patients.append(patient_id)
    
    # Function to copy all files from patients to split folders
    def copy_split(patient_list, split_name):
        total_images = 0
        for patient_id in patient_list:
            # Get all images for this patient
            patient_image_folder = os.path.join(image_path, patient_id)
            image_files = glob.glob(os.path.join(patient_image_folder, '*.jpg'))
            
            for image_file in image_files:
                base_name = os.path.splitext(os.path.basename(image_file))[0]
                
                # Copy image
                dst_image = os.path.join(output_path, split_name, 'images', f'{base_name}.jpg')
                shutil.copy2(image_file, dst_image)
                
                # Copy corresponding label
                src_label = os.path.join(label_path, patient_id, f'{base_name}.txt')
                dst_label = os.path.join(output_path, split_name, 'labels', f'{base_name}.txt')
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
                else:
                    print(f'Warning: Label file not found for patient {patient_id}, image {base_name}')
                
                total_images += 1
        
        return total_images
    
    # Copy files to respective splits
    train_count = copy_split(train_patients, 'train')
    val_count = copy_split(val_patients, 'val')
    test_count = copy_split(test_patients, 'test')
    
    # Print results
    print(f'Split complete (greedy algorithm by tumor type):')
    print(f'  Train: {len(train_patients)} patients ({train_count} images)')
    print(f'  Val: {len(val_patients)} patients ({val_count} images)')
    print(f'  Test: {len(test_patients)} patients ({test_count} images)')
    print(f'  Total: {len(patient_ids)} patients ({train_count + val_count + test_count} images)')
    print(f'\nPatient counts by tumor type:')
    print(f'  Test set: meningioma={test_patient_counts[0]}, glioma={test_patient_counts[1]}, pituitary_tumor={test_patient_counts[2]}')
    print(f'  Val set: meningioma={val_patient_counts[0]}, glioma={val_patient_counts[1]}, pituitary_tumor={val_patient_counts[2]}')
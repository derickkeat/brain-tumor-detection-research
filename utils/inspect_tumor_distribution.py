#!/usr/bin/env python3
"""
Script to inspect output/labels to get the distribution of 3 different types of tumors.
Counts all tumors across all label files in the output/labels directory.
"""

import os
from pathlib import Path
from collections import Counter
import sys

# Tumor type mapping
TUMOR_TYPES = {
    0: 'Meningioma',
    1: 'Glioma',
    2: 'Pituitary Tumor'
}

def inspect_tumor_distribution(labels_dir):
    """
    Read all label files in output/labels and analyze tumor type distribution.
    
    Args:
        labels_dir: Path to the labels directory (e.g., '../output/labels')
    """
    labels_path = Path(labels_dir)
    
    if not labels_path.exists():
        print(f"Error: Directory {labels_dir} does not exist")
        return
    
    # Find all .txt files recursively
    txt_files = list(labels_path.rglob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {labels_dir}")
        return
    
    print(f"Found {len(txt_files)} label files")
    print("Analyzing tumor distribution...\n")
    
    tumor_counts = Counter()
    files_with_errors = []
    files_with_no_tumors = []
    total_tumors = 0
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                file_has_tumors = False
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 1:
                            try:
                                class_id = int(parts[0])
                                if class_id in TUMOR_TYPES:
                                    tumor_counts[class_id] += 1
                                    total_tumors += 1
                                    file_has_tumors = True
                                else:
                                    files_with_errors.append((txt_file, f"Line {line_num}: Invalid class ID {class_id}"))
                            except ValueError:
                                files_with_errors.append((txt_file, f"Line {line_num}: Cannot parse class ID"))
                
                if not file_has_tumors and lines:
                    files_with_no_tumors.append(txt_file)
        except Exception as e:
            files_with_errors.append((txt_file, str(e)))
    
    # Display results
    print("=" * 70)
    print("TUMOR TYPE DISTRIBUTION")
    print("=" * 70)
    
    if not tumor_counts:
        print("No tumors found in label files!")
        return
    
    # Sort by class ID
    sorted_tumors = sorted(tumor_counts.items())
    total_samples = sum(tumor_counts.values())
    
    print(f"\nTotal tumors found: {total_samples}")
    print(f"Total label files: {len(txt_files)}")
    print(f"Label files with tumors: {len(txt_files) - len(files_with_no_tumors)}")
    print(f"Label files without tumors: {len(files_with_no_tumors)}")
    print("\n" + "-" * 70)
    print(f"{'Tumor Type':<20} | {'Class ID':<10} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 70)
    
    for class_id, count in sorted_tumors:
        tumor_name = TUMOR_TYPES.get(class_id, f"Unknown (ID: {class_id})")
        percentage = (count / total_samples) * 100
        print(f"{tumor_name:<20} | {class_id:<10} | {count:<10} | {percentage:>9.2f}%")
    
    # Calculate balance metrics
    print("\n" + "=" * 70)
    print("BALANCE ANALYSIS")
    print("=" * 70)
    
    counts = list(tumor_counts.values())
    if len(counts) > 1:
        min_count = min(counts)
        max_count = max(counts)
        avg_count = sum(counts) / len(counts)
        
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"Minimum tumor count: {min_count}")
        print(f"Maximum tumor count: {max_count}")
        print(f"Average tumor count: {avg_count:.2f}")
        print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2.0:
            print("\n⚠️  WARNING: Significant class imbalance detected!")
            print("   Consider using class weights or data augmentation.")
        elif imbalance_ratio > 1.5:
            print("\n⚠️  NOTE: Moderate class imbalance detected.")
        else:
            print("\n✓ Tumor types are relatively balanced.")
    
    # Show files with errors if any
    if files_with_errors:
        print("\n" + "=" * 70)
        print(f"FILES WITH ERRORS ({len(files_with_errors)})")
        print("=" * 70)
        for file_path, error in files_with_errors[:10]:  # Show first 10
            print(f"{file_path}: {error}")
        if len(files_with_errors) > 10:
            print(f"... and {len(files_with_errors) - 10} more")
    
    # Show files with no tumors if any
    if files_with_no_tumors:
        print("\n" + "=" * 70)
        print(f"FILES WITH NO TUMORS ({len(files_with_no_tumors)})")
        print("=" * 70)
        for file_path in files_with_no_tumors[:10]:  # Show first 10
            print(f"{file_path}")
        if len(files_with_no_tumors) > 10:
            print(f"... and {len(files_with_no_tumors) - 10} more")

if __name__ == "__main__":
    labels_dir = "../output/labels"
    
    if len(sys.argv) > 1:
        labels_dir = sys.argv[1]
    
    inspect_tumor_distribution(labels_dir)


#!/usr/bin/env python3
"""
Script to check data balance by reading the first word (class number) 
from all text files in output/btf/train/
"""

import os
from pathlib import Path
from collections import Counter
import sys

def analyze_data_balance(train_dir):
    """Read first word from all text files and analyze class distribution."""
    train_path = Path(train_dir)
    
    if not train_path.exists():
        print(f"Error: Directory {train_dir} does not exist")
        return
    
    # Find all .txt files recursively
    txt_files = list(train_path.rglob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {train_dir}")
        return
    
    print(f"Found {len(txt_files)} text files")
    print("Reading class labels...\n")
    
    class_counts = Counter()
    files_with_errors = []
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    first_word = first_line.split()[0]
                    try:
                        class_id = int(first_word)
                        class_counts[class_id] += 1
                    except (ValueError, IndexError):
                        files_with_errors.append((txt_file, "Invalid first word"))
                else:
                    files_with_errors.append((txt_file, "Empty file"))
        except Exception as e:
            files_with_errors.append((txt_file, str(e)))
    
    # Display results
    print("=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    
    if not class_counts:
        print("No valid class labels found!")
        return
    
    # Sort by class ID
    sorted_classes = sorted(class_counts.items())
    total_samples = sum(class_counts.values())
    
    print(f"\nTotal samples: {total_samples}")
    print(f"Number of classes: {len(class_counts)}")
    print("\nClass ID | Count    | Percentage")
    print("-" * 40)
    
    for class_id, count in sorted_classes:
        percentage = (count / total_samples) * 100
        print(f"{class_id:8d} | {count:8d} | {percentage:6.2f}%")
    
    # Calculate imbalance metrics
    print("\n" + "=" * 60)
    print("IMBALANCE ANALYSIS")
    print("=" * 60)
    
    counts = list(class_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    avg_count = sum(counts) / len(counts)
    
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"Minimum class count: {min_count}")
    print(f"Maximum class count: {max_count}")
    print(f"Average class count: {avg_count:.2f}")
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 2.0:
        print("\n⚠️  WARNING: Significant class imbalance detected!")
        print("   Consider using class weights or data augmentation.")
    elif imbalance_ratio > 1.5:
        print("\n⚠️  NOTE: Moderate class imbalance detected.")
    else:
        print("\n✓ Classes are relatively balanced.")
    
    # Show files with errors if any
    if files_with_errors:
        print("\n" + "=" * 60)
        print(f"FILES WITH ERRORS ({len(files_with_errors)})")
        print("=" * 60)
        for file_path, error in files_with_errors[:10]:  # Show first 10
            print(f"{file_path}: {error}")
        if len(files_with_errors) > 10:
            print(f"... and {len(files_with_errors) - 10} more")

if __name__ == "__main__":
    train_dir = "../output/btf/train"
    
    if len(sys.argv) > 1:
        train_dir = sys.argv[1]
    
    analyze_data_balance(train_dir)


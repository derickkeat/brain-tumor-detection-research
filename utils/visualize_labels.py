#!/usr/bin/env python3
"""
Script to fuse images and labels for visual inspection.
Verifies that images and labels are correctly processed from the original .mat files.
"""

import os
import cv2
import numpy as np
import argparse
import random
from pathlib import Path


class LabelVisualizer:
    """Visualizes YOLO format labels on images"""
    
    # Class names and colors
    CLASS_NAMES = {
        0: 'Meningioma',
        1: 'Glioma', 
        2: 'Pituitary Tumor'
    }
    
    CLASS_COLORS = {
        0: (255, 0, 0),      # Blue for Meningioma
        1: (0, 255, 0),      # Green for Glioma
        2: (0, 0, 255)       # Red for Pituitary Tumor
    }
    
    def __init__(self, output_dir='../output/btf', inspection_dir='./inspection_output'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Path to the btf output directory
            inspection_dir: Path to save inspection images
        """
        self.output_dir = Path(output_dir)
        self.inspection_dir = Path(inspection_dir)
        self.inspection_dir.mkdir(exist_ok=True, parents=True)
        
    def yolo_to_bbox(self, yolo_box, img_width, img_height):
        """
        Convert YOLO format (normalized) to pixel coordinates.
        
        Args:
            yolo_box: [center_x, center_y, width, height] (normalized 0-1)
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            (x1, y1, x2, y2) in pixel coordinates
        """
        center_x, center_y, width, height = yolo_box
        
        # Convert to pixel coordinates
        center_x_px = center_x * img_width
        center_y_px = center_y * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        # Calculate corners
        x1 = int(center_x_px - width_px / 2)
        y1 = int(center_y_px - height_px / 2)
        x2 = int(center_x_px + width_px / 2)
        y2 = int(center_y_px + height_px / 2)
        
        return (x1, y1, x2, y2)
    
    def draw_bounding_box(self, image, bbox, class_id, thickness=2):
        """
        Draw bounding box and label on image.
        
        Args:
            image: OpenCV image
            bbox: (x1, y1, x2, y2) in pixel coordinates
            class_id: Class ID (0, 1, or 2)
            thickness: Box thickness
        """
        x1, y1, x2, y2 = bbox
        color = self.CLASS_COLORS.get(class_id, (255, 255, 255))
        class_name = self.CLASS_NAMES.get(class_id, f'Class {class_id}')
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        label = f'{class_name}'
        
        # Get text size for background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
        
    def visualize_sample(self, image_path, label_path, save_path=None):
        """
        Visualize a single image with its label.
        
        Args:
            image_path: Path to image file
            label_path: Path to label file
            save_path: Path to save the visualization (optional)
            
        Returns:
            Annotated image
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None
            
        img_height, img_width = image.shape[:2]
        
        # Read label file
        if not os.path.exists(label_path):
            print(f"Warning: No label file found at {label_path}")
            # Still save the image without annotations
            if save_path:
                cv2.imwrite(str(save_path), image)
            return image
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # Process each bounding box
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Warning: Invalid label format in {label_path}: {line.strip()}")
                continue
                
            try:
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert to pixel coordinates
                bbox = self.yolo_to_bbox(
                    [center_x, center_y, width, height],
                    img_width,
                    img_height
                )
                
                # Draw bounding box
                self.draw_bounding_box(image, bbox, class_id)
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing line '{line.strip()}': {e}")
                continue
        
        # Add filename as title
        title = os.path.basename(image_path)
        cv2.putText(
            image,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Save if path provided
        if save_path:
            cv2.imwrite(str(save_path), image)
            print(f"Saved visualization to {save_path}")
            
        return image
    
    def visualize_random_samples(self, split='train', num_samples=10):
        """
        Visualize random samples from a dataset split.
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            num_samples: Number of samples to visualize
        """
        print(f"\n{'='*70}")
        print(f"VISUALIZING {num_samples} RANDOM SAMPLES FROM {split.upper()} SET")
        print(f"{'='*70}\n")
        
        # Get paths
        images_dir = self.output_dir / split / 'images'
        labels_dir = self.output_dir / split / 'labels'
        
        if not images_dir.exists():
            print(f"Error: Images directory not found: {images_dir}")
            return
            
        # Get all image files
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        if not image_files:
            print(f"Error: No images found in {images_dir}")
            return
            
        # Select random samples
        num_samples = min(num_samples, len(image_files))
        selected_images = random.sample(image_files, num_samples)
        
        # Create output directory for this split
        split_output_dir = self.inspection_dir / split
        split_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Visualize each sample
        for idx, img_path in enumerate(selected_images, 1):
            label_path = labels_dir / (img_path.stem + '.txt')
            save_path = split_output_dir / f"sample_{idx:03d}_{img_path.name}"
            
            print(f"[{idx}/{num_samples}] Processing: {img_path.name}")
            self.visualize_sample(img_path, label_path, save_path)
            
        print(f"\nSaved {num_samples} visualizations to {split_output_dir}")
        
    def visualize_specific_patient(self, patient_id):
        """
        Visualize all slices from a specific patient.
        
        Args:
            patient_id: Patient ID (e.g., '100360' or 'MR017260F')
        """
        print(f"\n{'='*70}")
        print(f"VISUALIZING ALL SLICES FOR PATIENT {patient_id}")
        print(f"{'='*70}\n")
        
        images_dir = Path(f'../output/images/{patient_id}')
        labels_dir = Path(f'../output/labels/{patient_id}')
        
        if not images_dir.exists():
            print(f"Error: Patient directory not found: {images_dir}")
            return
            
        # Get all image files
        image_files = sorted(
            list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')),
            key=lambda x: int(''.join(filter(str.isdigit, x.stem)) or '0')
        )
        
        if not image_files:
            print(f"Error: No images found for patient {patient_id}")
            return
            
        # Create output directory for this patient
        patient_output_dir = self.inspection_dir / f"patient_{patient_id}"
        patient_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Visualize each slice
        for idx, img_path in enumerate(image_files, 1):
            label_path = labels_dir / (img_path.stem + '.txt')
            save_path = patient_output_dir / f"{img_path.name}"
            
            print(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")
            self.visualize_sample(img_path, label_path, save_path)
            
        print(f"\nSaved {len(image_files)} visualizations to {patient_output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize brain tumor images with bounding box labels'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test', 'all'],
        default='train',
        help='Dataset split to visualize (default: train)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of random samples to visualize per split (default: 10)'
    )
    parser.add_argument(
        '--patient-id',
        type=str,
        help='Visualize all slices from a specific patient ID'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../output/btf',
        help='Path to btf output directory (default: ../output/btf)'
    )
    parser.add_argument(
        '--inspection-dir',
        type=str,
        default='./inspection_output',
        help='Path to save inspection images (default: ./inspection_output)'
    )
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = LabelVisualizer(
        output_dir=args.output_dir,
        inspection_dir=args.inspection_dir
    )
    
    # Visualize specific patient if requested
    if args.patient_id:
        visualizer.visualize_specific_patient(args.patient_id)
        return
    
    # Visualize splits
    if args.split == 'all':
        for split in ['train', 'val', 'test']:
            visualizer.visualize_random_samples(split, args.num_samples)
    else:
        visualizer.visualize_random_samples(args.split, args.num_samples)
    
    print(f"\n{'='*70}")
    print("INSPECTION COMPLETE")
    print(f"{'='*70}")
    print(f"All visualizations saved to: {visualizer.inspection_dir}")
    print("\nLegend:")
    print("  Blue = Meningioma")
    print("  Green = Glioma")
    print("  Red = Pituitary Tumor")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()


import os
import glob
import argparse

from image import mat_to_jpg
from label import mat_to_yolo_label
from split import split_data, split_data_real_world_distribution, split_data_preserve_distribution

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the data')
    parser.add_argument('--rwd', action='store_true', help='Use real-world distribution (rwd) split')
    parser.add_argument('--stratified', action='store_true', help='Use stratified split')
    args = parser.parse_args()

    # Set your input and output paths
    input_folders = ['../dataset/brainTumorDataPublic_1-766', '../dataset/brainTumorDataPublic_767-1532', '../dataset/brainTumorDataPublic_1533-2298', '../dataset/brainTumorDataPublic_2299-3064']
    output_folder_images = '../output/images/'
    output_folder_labels = '../output/labels/'
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder_images):
        os.makedirs(output_folder_images)
    if not os.path.exists(output_folder_labels):
        os.makedirs(output_folder_labels)
    
    for input_folder in input_folders:
        # Get list of all .mat files in input folder
        file_list = glob.glob(os.path.join(input_folder, '*.mat'))
        
        for file_path in file_list:
            mat_to_jpg(file_path, output_folder_images)
            mat_to_yolo_label(file_path, output_folder_labels)
    print('Processing complete!')

    # Split the data into train/val/test sets
    print('\nSplitting data into train/val/test sets...')
    if args.rwd:
        split_data_real_world_distribution(output_folder_images, output_folder_labels, '../output/btf_rwd/')  # 70% train, 10% val, 20% test
    elif args.stratified:
        split_data_preserve_distribution(output_folder_images, output_folder_labels, '../output/btf_stratified/', train_ratio=0.8, val_ratio=0.0)  # 70% train, 10% val, 20% test
    else:
        split_data(output_folder_images, output_folder_labels, '../output/btf/', train_ratio=0.7, val_ratio=0.1)  # 70% train, 10% val, 20% test

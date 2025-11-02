import os
import glob

from image import mat_to_jpg

if __name__ == '__main__':
    # Set your input and output paths
    input_folders = ['../dataset/brainTumorDataPublic_1-766', '../dataset/brainTumorDataPublic_767-1532', '../dataset/brainTumorDataPublic_1533-2298', '../dataset/brainTumorDataPublic_2299-3064']
    output_folder = '../output/images/'
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for input_folder in input_folders:
        # Get list of all .mat files in input folder
        file_list = glob.glob(os.path.join(input_folder, '*.mat'))
        
        for file_path in file_list:
            mat_to_jpg(file_path, output_folder)
    print('Processing complete!')

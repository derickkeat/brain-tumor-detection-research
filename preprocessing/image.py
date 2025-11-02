import numpy as np
import h5py
from PIL import Image

import os
import glob

def mat_to_jpg(file_path, output_folder):
    """
    Process .mat files containing image data and labels.
    Normalizes images to uint8 range and saves them as JPEGs organized by label.
    
    Args:
        file_path: Path to .mat file
        output_folder: Path to folder where processed images will be saved
    """

    # Load .mat file (MATLAB v7.3 uses HDF5 format)
    with h5py.File(file_path, 'r') as mat_data:
        cjdata = mat_data['cjdata']
        
        # Extract image and label from cjdata structure
        # Image is stored directly as a 512x512 array
        im1 = np.array(cjdata['image'], dtype=np.float64)
        
        # Label is stored as a 1x1 array
        label = int(cjdata['label'][0, 0])
        
        # Normalize image to 0-255 range
        min1 = im1.min()
        max1 = im1.max()
        im = np.uint8(255 / (max1 - min1) * (im1 - min1))
        
        # Convert to PIL Image and resize to 224x224
        img = Image.fromarray(im)
        img = img.resize((224, 224), Image.LANCZOS)
        
        # Create label folder if it doesn't exist
        label_folder = os.path.join(output_folder, str(label))
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        
        # Get base filename without extension
        file_name = os.path.basename(file_path)
        file_name_base = os.path.splitext(file_name)[0]
        
        # Save image as JPEG
        output_file_path = os.path.join(label_folder, f'{file_name_base}.jpg')
        img.save(output_file_path)
        
        print(f'Processed: {file_name} -> {output_file_path}')
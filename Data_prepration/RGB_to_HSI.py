import os
import re
import ast
import cv2
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import albumentations as A

def convert_to_hsi_and_clahe(image):
    # Convert image to HSI format
    img_hsi = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # Apply CLAHE
    transform = A.Compose([
        A.augmentations.transforms.CLAHE(clip_limit=4.0,
                                          tile_grid_size=(8, 8),
                                          always_apply=False,
                                          p=1.0)
    ])
    transformed = transform(image=img_hsi)
    img_clahe = transformed['image']

    return img_clahe

# Input folder containing the images
input_folder = '/Users/asad/Documents/pcbdata/pcb_image/'

# Output folder for processed images
output_folder = '/Users/asad/Documents/pcbdata/pcb_HSI_images/'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'): # Adjust the file extensions as needed
        # Read the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Apply the function
        processed_image = convert_to_hsi_and_clahe(image)

        # Specify the output path for the processed image
        output_path = os.path.join(output_folder, filename)

        # Save the processed image
        cv2.imwrite(output_path, cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        print(f"Processed image saved to {output_path}")

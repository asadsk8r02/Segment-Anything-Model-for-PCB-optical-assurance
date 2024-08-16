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

def delete_black_masks(image_folder, mask_folder):
    image_files = os.listdir(image_folder)
    mask_files = os.listdir(mask_folder)

    for image_file in image_files:
        image_name, _ = os.path.splitext(image_file)
        mask_name = image_name + "_test_smd_mask.png"

        if mask_name not in mask_files:
            # print(f"No matching mask found for image: {image_file}")
            continue

        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, mask_name)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            # print(f"Failed to read mask: {mask_path}")
            continue

        if cv2.countNonZero(mask) == 0:
            os.remove(image_path)
            os.remove(mask_path)
            # print(f"Deleted image: {image_path}")
            # print(f"Deleted mask: {mask_path}")

image_folder = "/Users/asad/Documents/pcbdata/pcb_images_SAM_BTN"
mask_folder = results_dir = "/Users/asad/Documents/pcbdata/Opaque_zero_mask_BTN_only"
delete_black_masks(image_folder, mask_folder)


### Opaque_zero_mask_BTN_only and pro_pcb_images_BTN are renamed as masks_BTN and images_BTN respectively

image_folder = "/Users/asad/Documents/pcbdata/Unet_single_component/images_BTN"
mask_folder = "/Users/asad/Documents/pcbdata/Unet_single_component/masks_BTN"

image_filenames = [f.split('.png')[0] for f in os.listdir(image_folder) if f.endswith('.png')]

mask_filenames = [f.split('_test_smd_mask.png')[0] for f in os.listdir(mask_folder) if f.endswith('_test_smd_mask.png')]

for image_filename in image_filenames:
    if image_filename not in mask_filenames:
        image_path = os.path.join(image_folder, f"{image_filename}.png")
        os.remove(image_path)
        # print(f"Deleted {image_path}")



image_filenames = [f.split('.png')[0] for f in os.listdir(image_folder) if f.endswith('.png')]

# Extract the base filenames for masks, assuming the format 'pcb_0f_cc_11_test_smd_mask.png'
mask_filenames = [f.split('_test_smd_mask.png')[0] for f in os.listdir(mask_folder) if f.endswith('_test_smd_mask.png')]

# Check for masks without corresponding images and remove them
for mask_filename in mask_filenames:
    if mask_filename not in image_filenames:
        mask_path = os.path.join(mask_folder, f"{mask_filename}_test_smd_mask.png")
        os.remove(mask_path)
        # Uncomment the next line to see which files are being deleted
        # print(f"Deleted {mask_path}")


# Function to create directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Paths to the folders containing images and masks
image_folder = "/Users/asad/Documents/pcbdata/Unet_single_component/images_BTN"
mask_folder = "/Users/asad/Documents/pcbdata/Unet_single_component/masks_BTN"

# Output folders for sub-images and sub-masks
output_image_folder = "/Users/asad/Documents/pcbdata/Unet_single_component/SAM_BTN/images"
output_mask_folder = "/Users/asad/Documents/pcbdata/Unet_single_component/SAM_BTN/masks"

# Create output directories if they don't exist
create_directory(output_image_folder)
create_directory(output_mask_folder)

# Iterate over each file in the image folder
for image_filename in os.listdir(image_folder):
    # Check if the file is a PNG image
    if image_filename.endswith(".png"):
        # Extract the corresponding mask filename
        mask_filename = image_filename.replace(".png", "_test_smd_mask.png")
        
        # Load the large image and its mask
        large_image = cv2.imread(os.path.join(image_folder, image_filename))
        mask = cv2.imread(os.path.join(mask_folder, mask_filename), cv2.IMREAD_GRAYSCALE)
        
        # Find contours in the mask image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Iterate over each contour
        for i, contour in enumerate(contours):
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Expand the bounding box by 50 pixels in each direction
            x -= 50
            y -= 50
            w += 100
            h += 100
            
            # Ensure the bounding box is within the image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(large_image.shape[1] - x, w)
            h = min(large_image.shape[0] - y, h)
            
            # Crop the corresponding region from both the large image and the mask image
            sub_image = large_image[y:y+h, x:x+w]
            sub_mask = mask[y:y+h, x:x+w]
            
            # Construct the base filename without extension
            base_filename = os.path.splitext(image_filename)[0]
            
            # Save the cropped region as a sub-image along with its corresponding mask
            image_output_path = os.path.join(output_image_folder, f"{base_filename}_{i}_image.png")
            mask_output_path = os.path.join(output_mask_folder, f"{base_filename}_{i}_mask.png")
            
            cv2.imwrite(image_output_path, sub_image)
            cv2.imwrite(mask_output_path, sub_mask)


def contains_multiple_objects(sub_mask_path):
    sub_mask = cv2.imread(sub_mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(sub_mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, _ = cv2.connectedComponents(binary_mask)
    return num_labels - 1 > 1

# Directory paths
sub_images_directory = "/Users/asad/Documents/pcbdata/Unet_single_component/SAM_BTN/images"
sub_masks_directory = "/Users/asad/Documents/pcbdata/Unet_single_component/SAM_BTN/masks"

# Check each sub-mask for multiple objects and delete corresponding sub-images and sub-masks
for filename in os.listdir(sub_masks_directory):
    if filename.endswith(".png"):
        sub_mask_path = os.path.join(sub_masks_directory, filename)
        sub_image_path = os.path.join(sub_images_directory, filename.replace("_mask.png", "_image.png"))
        if contains_multiple_objects(sub_mask_path):
            os.remove(sub_mask_path)
            os.remove(sub_image_path)
#             print(f"Deleted {filename} and corresponding sub-image")


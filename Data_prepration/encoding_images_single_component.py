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

label_seed = 42
color_seed = 123

random.seed(label_seed)
np.random.seed(color_seed)

image_folder = "/Users/asad/Documents/pcbdata/pcb_image"
metadata_folder = "/Users/asad/Documents/pcbdata/new_merged_removed_final"
results_dir = "/Users/asad/Documents/pcbdata/Opaque_zero_mask_R_only"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    metadata_file = os.path.splitext(image_file)[0] + '_smd.csv'
    metadata_path = os.path.join(metadata_folder, metadata_file)

    if os.path.isfile(metadata_path):
                      
        image = cv2.imread(image_path)
        smd_metadata = pd.read_csv(metadata_path)

        smd_annotation = smd_metadata["Vertices"].fillna('[]').apply(ast.literal_eval)
        coordinates = [[tuple(coord) for coord in inner_list] for sublist in smd_annotation for inner_list in sublist]

        Designator = smd_metadata["Designator"]

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for designator, coords in zip(Designator, coordinates):
            if designator == "R":
                cv2.fillPoly(mask, [np.array(coords)], 255)
                      
        result_path = os.path.join(results_dir, os.path.splitext(image_file)[0] + '_test_smd_mask.png')
        cv2.imwrite(result_path, mask)

def sliding_window(image, window_size, stride):
    height, width, _ = image.shape
    window_height, window_width = window_size

    for y in range(0, height - window_height + 1, stride):
        for x in range(0, width - window_width + 1, stride):
            crop_image = image[y:y+window_height, x:x+window_width]
            yield crop_image, (x, y)

def save_cropped_images(image_folder, mask_folder, window_size, stride, output_folder):
    image_files = os.listdir(image_folder)
    mask_files = os.listdir(mask_folder)

    image_output_folder = os.path.join(output_folder, "cropped_images")
    mask_output_folder = os.path.join(output_folder, "cropped_masks")
    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(mask_output_folder, exist_ok=True)

    for image_file in image_files:
        image_name, image_ext = os.path.splitext(image_file)
        pattern = f"^{re.escape(image_name)}_.*{re.escape(image_ext)}$"
        matching_mask_files = [mask_file for mask_file in mask_files if re.match(pattern, mask_file, re.IGNORECASE)]
        
        if len(matching_mask_files) == 0:
#             print(f"No matching mask found for image: {image_file}")
            continue

        image_path = os.path.join(image_folder, image_file)

#         print(f"Processing image: {image_path}")

        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to read image: {image_path}")
            continue

        for i, mask_file in enumerate(matching_mask_files):
            mask_path = os.path.join(mask_folder, mask_file)

#             print(f"Processing mask: {mask_path}")

            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            if mask is None:
#                 print(f"Failed to read mask: {mask_path}")
                continue

            for j, (crop_image, (x, y)) in enumerate(sliding_window(image, window_size, stride)):
                crop_mask = mask[y:y+window_size[0], x:x+window_size[1]]
                
                crop_image_path = os.path.join(image_output_folder, f"crop_{image_name}_{i}_{j}_image.jpg")
                crop_mask_path = os.path.join(mask_output_folder, f"crop_{image_name}_{i}_{j}_mask.png")
                cv2.imwrite(crop_image_path, crop_image)
                cv2.imwrite(crop_mask_path, crop_mask)

#                 print(f"Saved cropped image: {crop_image_path}")
#                 print(f"Saved cropped mask: {crop_mask_path}")

image_folder = "/Users/asad/Documents/pcbdata/pcb_image"
mask_folder = "/Users/asad/Documents/pcbdata/Opaque_zero_mask_R_only"
window_size = (256, 256)
stride = 128
output_folder = "/Users/asad/Documents/pcbdata/full_sub_images_opaque_new_R_only"

save_cropped_images(image_folder, mask_folder, window_size, stride, output_folder)

def delete_black_masks(image_folder, mask_folder):
    image_files = os.listdir(image_folder)
    mask_files = os.listdir(mask_folder)

    for image_file in image_files:
        image_name, _ = os.path.splitext(image_file)
        mask_name = image_name.replace("_image", "_mask") + ".png"

        if mask_name not in mask_files:
#             print(f"No matching mask found for image: {image_file}")
            continue

        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, mask_name)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
#             print(f"Failed to read mask: {mask_path}")
            continue

        if cv2.countNonZero(mask) == 0:
            os.remove(image_path)
            os.remove(mask_path)
#             print(f"Deleted image: {image_path}")
#             print(f"Deleted mask: {mask_path}")

image_folder = "/Users/asad/Documents/pcbdata/full_sub_images_opaque_new_R_only/cropped_images"
mask_folder = "/Users/asad/Documents/pcbdata/full_sub_images_opaque_new_R_only/cropped_masks"

delete_black_masks(image_folder, mask_folder)

mask_path ="/Users/asad/Documents/pcbdata/full_sub_images_opaque_new_R_only/cropped_masks/crop_pcb_7b_cc_7_0_103_mask.png"
image = Image.open(mask_path)
mask_rgb = np.array(image)
mask_rgb.shape


# Cross-checking if the images and their corresponding annotation files exists.
import os

image_folder = "/Users/asad/Documents/pcbdata/full_sub_images_opaque_new_R_only/cropped_images"
mask_folder = "/Users/asad/Documents/pcbdata/full_sub_images_opaque_new_R_only/cropped_masks"

image_filenames = [f.split('_image')[0] for f in os.listdir(image_folder) if f.endswith('_image.jpg')]

mask_filenames = [f.split('_mask')[0] for f in os.listdir(mask_folder) if f.endswith('_mask.png')]

for image_filename in image_filenames:
    if image_filename not in mask_filenames:
        image_path = os.path.join(images_folder, f"{image_filename}_image.jpg")
        os.remove(image_path)
        print(f"Deleted {image_path}")

print("Done")
import os

image_folder = "/Users/asad/Documents/pcbdata/full_sub_images_opaque_new_R_only/cropped_images"
mask_folder = "/Users/asad/Documents/pcbdata/full_sub_images_opaque_new_R_only/cropped_masks"

image_filenames = [f.split('_image')[0] for f in os.listdir(image_folder) if f.endswith('_image.jpg')]
mask_filenames = [f.split('_mask')[0] for f in os.listdir(mask_folder) if f.endswith('_mask.png')]

for mask_filename in mask_filenames:
    if mask_filename not in image_filenames:
        mask_path = os.path.join(masks_folder, f"{mask_filename}_mask.png")
        os.remove(mask_path)
        print(f"Deleted {mask_path}")

print("Done")

# Density slicing operation.

# For 5% percent
import os
import cv2
import numpy as np

folder_path = "/Users/asad/Documents/pcbdata/full_sub_images_opaque_new_R_only/cropped_masks_5"

for filename in os.listdir(folder_path):
    if filename.endswith(".png"):  
        mask_path = os.path.join(folder_path, filename)
        test_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        val, counts = np.unique(test_mask, return_counts=True)

        non_zero_percentage = 1 - (counts[0] / counts.sum())

        if non_zero_percentage > 0.05:
            pass
        else:
            os.remove(mask_path)
            #print(f"Deleted {filename}")

print("Done")


# Cross-checking if the images and their corresponding annotation files exists.
import os

image_folder = "/Users/asad/Documents/pcbdata/full_sub_images_opaque_new_R_only/cropped_images_5"
mask_folder = "/Users/asad/Documents/pcbdata/full_sub_images_opaque_new_R_only/cropped_masks_5"

image_filenames = [f.split('_image')[0] for f in os.listdir(image_folder) if f.endswith('_image.jpg')]

mask_filenames = [f.split('_mask')[0] for f in os.listdir(mask_folder) if f.endswith('_mask.png')]

for image_filename in image_filenames:
    if image_filename not in mask_filenames:
        image_path = os.path.join(image_folder, f"{image_filename}_image.jpg")
        os.remove(image_path)
#         print(f"Deleted {image_path}")

print("Done")
import os

image_folder = "/Users/asad/Documents/pcbdata/full_sub_images_opaque_new_R_only/cropped_images_5"
mask_folder = "/Users/asad/Documents/pcbdata/full_sub_images_opaque_new_R_only/cropped_masks_5"

image_filenames = [f.split('_image')[0] for f in os.listdir(image_folder) if f.endswith('_image.jpg')]
mask_filenames = [f.split('_mask')[0] for f in os.listdir(mask_folder) if f.endswith('_mask.png')]

for mask_filename in mask_filenames:
    if mask_filename not in image_filenames:
        mask_path = os.path.join(masks_folder, f"{mask_filename}_mask.png")
        os.remove(mask_path)
#         print(f"Deleted {mask_path}")

print("Done")

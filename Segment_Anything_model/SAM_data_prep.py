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

class FileManager:
    @staticmethod
    def create_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def delete_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
            # Uncomment the next line to see which files are being deleted
            # print(f"Deleted {file_path}")

class MaskManager:
    def __init__(self, image_folder, mask_folder):
        self.image_folder = image_folder
        self.mask_folder = mask_folder

    def delete_black_masks(self):
        image_files = os.listdir(self.image_folder)
        mask_files = os.listdir(self.mask_folder)

        for image_file in image_files:
            image_name, _ = os.path.splitext(image_file)
            mask_name = f"{image_name}_test_smd_mask.png"

            if mask_name not in mask_files:
                continue

            image_path = os.path.join(self.image_folder, image_file)
            mask_path = os.path.join(self.mask_folder, mask_name)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None or cv2.countNonZero(mask) == 0:
                FileManager.delete_file(image_path)
                FileManager.delete_file(mask_path)

    def clean_unmatched_files(self):
        image_filenames = [f.split('.png')[0] for f in os.listdir(self.image_folder) if f.endswith('.png')]
        mask_filenames = [f.split('_test_smd_mask.png')[0] for f in os.listdir(self.mask_folder) if f.endswith('_test_smd_mask.png')]

        # Delete images without corresponding masks
        for image_filename in image_filenames:
            if image_filename not in mask_filenames:
                image_path = os.path.join(self.image_folder, f"{image_filename}.png")
                FileManager.delete_file(image_path)

        # Delete masks without corresponding images
        for mask_filename in mask_filenames:
            if mask_filename not in image_filenames:
                mask_path = os.path.join(self.mask_folder, f"{mask_filename}_test_smd_mask.png")
                FileManager.delete_file(mask_path)

class ImageProcessor:
    def __init__(self, image_folder, mask_folder, output_image_folder, output_mask_folder):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.output_image_folder = output_image_folder
        self.output_mask_folder = output_mask_folder
        FileManager.create_directory(self.output_image_folder)
        FileManager.create_directory(self.output_mask_folder)

    def process_images(self):
        for image_filename in os.listdir(self.image_folder):
            if image_filename.endswith(".png"):
                mask_filename = image_filename.replace(".png", "_test_smd_mask.png")
                large_image = cv2.imread(os.path.join(self.image_folder, image_filename))
                mask = cv2.imread(os.path.join(self.mask_folder, mask_filename), cv2.IMREAD_GRAYSCALE)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                self._save_sub_images(image_filename, large_image, mask, contours)

    def _save_sub_images(self, image_filename, large_image, mask, contours):
        base_filename = os.path.splitext(image_filename)[0]

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = self._expand_bounding_box(x, y, w, h, large_image.shape)

            sub_image = large_image[y:y+h, x:x+w]
            sub_mask = mask[y:y+h, x:x+w]

            image_output_path = os.path.join(self.output_image_folder, f"{base_filename}_{i}_image.png")
            mask_output_path = os.path.join(self.output_mask_folder, f"{base_filename}_{i}_mask.png")

            cv2.imwrite(image_output_path, sub_image)
            cv2.imwrite(mask_output_path, sub_mask)

    @staticmethod
    def _expand_bounding_box(x, y, w, h, image_shape):
        x = max(0, x - 50)
        y = max(0, y - 50)
        w = min(image_shape[1] - x, w + 100)
        h = min(image_shape[0] - y, h + 100)
        return x, y, w, h

class ObjectChecker:
    def __init__(self, sub_images_directory, sub_masks_directory):
        self.sub_images_directory = sub_images_directory
        self.sub_masks_directory = sub_masks_directory

    def check_and_delete_multiple_objects(self):
        for filename in os.listdir(self.sub_masks_directory):
            if filename.endswith(".png"):
                sub_mask_path = os.path.join(self.sub_masks_directory, filename)
                sub_image_path = os.path.join(self.sub_images_directory, filename.replace("_mask.png", "_image.png"))
                if self.contains_multiple_objects(sub_mask_path):
                    FileManager.delete_file(sub_mask_path)
                    FileManager.delete_file(sub_image_path)

    @staticmethod
    def contains_multiple_objects(sub_mask_path):
        sub_mask = cv2.imread(sub_mask_path, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(sub_mask, 127, 255, cv2.THRESH_BINARY)
        num_labels, _ = cv2.connectedComponents(binary_mask)
        return num_labels - 1 > 1


def main():
    # Paths
    image_folder = "/Users/asad/Documents/pcbdata/Unet_single_component/images_BTN"
    mask_folder = "/Users/asad/Documents/pcbdata/Unet_single_component/masks_BTN"
    output_image_folder = "/Users/asad/Documents/pcbdata/Unet_single_component/SAM_BTN/images"
    output_mask_folder = "/Users/asad/Documents/pcbdata/Unet_single_component/SAM_BTN/masks"

    # Step 1: Delete black masks
    mask_manager = MaskManager(image_folder, mask_folder)
    mask_manager.delete_black_masks()
    mask_manager.clean_unmatched_files()

    # Step 2: Process images and generate sub-images and sub-masks
    image_processor = ImageProcessor(image_folder, mask_folder, output_image_folder, output_mask_folder)
    image_processor.process_images()

    # Step 3: Check for multiple objects in sub-masks and delete corresponding files
    object_checker = ObjectChecker(output_image_folder, output_mask_folder)
    object_checker.check_and_delete_multiple_objects()


if __name__ == "__main__":
    main()

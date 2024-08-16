import os
import re
import ast
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder


class MaskCreator:
    def __init__(self, image_folder, metadata_folder, results_dir):
        self.image_folder = image_folder
        self.metadata_folder = metadata_folder
        self.results_dir = results_dir
        self._create_directory(self.results_dir)

    def create_masks(self):
        image_files = [f for f in os.listdir(self.image_folder) if f.endswith('.png')]

        for image_file in image_files:
            image_path = os.path.join(self.image_folder, image_file)
            metadata_file = os.path.splitext(image_file)[0] + '_smd.csv'
            metadata_path = os.path.join(self.metadata_folder, metadata_file)

            if os.path.isfile(metadata_path):
                self._process_image(image_path, metadata_path, image_file)

    def _process_image(self, image_path, metadata_path, image_file):
        image = cv2.imread(image_path)
        smd_metadata = pd.read_csv(metadata_path)

        smd_annotation = smd_metadata["Vertices"].fillna('[]').apply(ast.literal_eval)
        coordinates = [[tuple(coord) for coord in inner_list] for sublist in smd_annotation for inner_list in sublist]
        Designator = smd_metadata["Designator"]

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for designator, coords in zip(Designator, coordinates):
            if designator == "C":
                cv2.fillPoly(mask, [np.array(coords)], 255)

        result_path = os.path.join(self.results_dir, os.path.splitext(image_file)[0] + '_test_smd_mask.png')
        cv2.imwrite(result_path, mask)

    @staticmethod
    def _create_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


class ImageCropper:
    def __init__(self, image_folder, mask_folder, window_size, stride, output_folder):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.window_size = window_size
        self.stride = stride
        self.output_folder = output_folder
        self.image_output_folder = os.path.join(self.output_folder, "cropped_images")
        self.mask_output_folder = os.path.join(self.output_folder, "cropped_masks")
        MaskCreator._create_directory(self.image_output_folder)
        MaskCreator._create_directory(self.mask_output_folder)

    def save_cropped_images(self):
        image_files = os.listdir(self.image_folder)
        mask_files = os.listdir(self.mask_folder)

        for image_file in image_files:
            image_name, image_ext = os.path.splitext(image_file)
            pattern = f"^{re.escape(image_name)}_.*{re.escape(image_ext)}$"
            matching_mask_files = [mask_file for mask_file in mask_files if re.match(pattern, mask_file, re.IGNORECASE)]

            if len(matching_mask_files) == 0:
                continue

            image_path = os.path.join(self.image_folder, image_file)
            image = cv2.imread(image_path)

            if image is None:
                continue

            for i, mask_file in enumerate(matching_mask_files):
                mask_path = os.path.join(self.mask_folder, mask_file)
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

                if mask is None:
                    continue

                for j, (crop_image, (x, y)) in enumerate(self.sliding_window(image)):
                    crop_mask = mask[y:y+self.window_size[0], x:x+self.window_size[1]]
                    crop_image_path = os.path.join(self.image_output_folder, f"crop_{image_name}_{i}_{j}_image.jpg")
                    crop_mask_path = os.path.join(self.mask_output_folder, f"crop_{image_name}_{i}_{j}_mask.png")
                    cv2.imwrite(crop_image_path, crop_image)
                    cv2.imwrite(crop_mask_path, crop_mask)

    def sliding_window(self, image):
        height, width, _ = image.shape
        window_height, window_width = self.window_size

        for y in range(0, height - window_height + 1, self.stride):
            for x in range(0, width - window_width + 1, self.stride):
                crop_image = image[y:y+window_height, x:x+window_width]
                yield crop_image, (x, y)


class MaskCleaner:
    def __init__(self, image_folder, mask_folder):
        self.image_folder = image_folder
        self.mask_folder = mask_folder

    def delete_black_masks(self):
        image_files = os.listdir(self.image_folder)
        mask_files = os.listdir(self.mask_folder)

        for image_file in image_files:
            image_name, _ = os.path.splitext(image_file)
            mask_name = image_name.replace("_image", "_mask") + ".png"

            if mask_name not in mask_files:
                continue

            image_path = os.path.join(self.image_folder, image_file)
            mask_path = os.path.join(self.mask_folder, mask_name)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                continue

            if cv2.countNonZero(mask) == 0:
                os.remove(image_path)
                os.remove(mask_path)


class ImageMaskSynchronizer:
    def __init__(self, image_folder, mask_folder):
        self.image_folder = image_folder
        self.mask_folder = mask_folder

    def synchronize_images_masks(self):
        self._delete_unmatched_images()
        self._delete_unmatched_masks()

    def _delete_unmatched_images(self):
        image_filenames = [f.split('_image')[0] for f in os.listdir(self.image_folder) if f.endswith('_image.jpg')]
        mask_filenames = [f.split('_mask')[0] for f in os.listdir(self.mask_folder) if f.endswith('_mask.png')]

        for image_filename in image_filenames:
            if image_filename not in mask_filenames:
                image_path = os.path.join(self.image_folder, f"{image_filename}_image.jpg")
                os.remove(image_path)
                print(f"Deleted {image_path}")

    def _delete_unmatched_masks(self):
        mask_filenames = [f.split('_mask')[0] for f in os.listdir(self.mask_folder) if f.endswith('_mask.png')]
        image_filenames = [f.split('_image')[0] for f in os.listdir(self.image_folder) if f.endswith('_image.jpg')]

        for mask_filename in mask_filenames:
            if mask_filename not in image_filenames:
                mask_path = os.path.join(self.mask_folder, f"{mask_filename}_mask.png")
                os.remove(mask_path)
                print(f"Deleted {mask_path}")


class MaskFilter:
    def __init__(self, mask_folder):
        self.mask_folder = mask_folder

    def filter_masks(self, threshold=0.05):
        for filename in os.listdir(self.mask_folder):
            if filename.endswith(".png"):
                mask_path = os.path.join(self.mask_folder, filename)
                test_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                val, counts = np.unique(test_mask, return_counts=True)
                non_zero_percentage = 1 - (counts[0] / counts.sum())

                if non_zero_percentage <= threshold:
                    os.remove(mask_path)
                    print(f"Deleted {filename}")


def main():
    # Set seeds
    label_seed = 42
    color_seed = 123
    random.seed(label_seed)
    np.random.seed(color_seed)

    # Paths
    image_folder = "/Users/asad/Documents/pcbdata/pcb_image"
    metadata_folder = "/Users/asad/Documents/pcbdata/new_merged_removed_final"
    results_dir = "/Users/asad/Documents/pcbdata/Opaque_zero_mask_C_only"

    # Step 1: Create masks
    mask_creator = MaskCreator(image_folder, metadata_folder, results_dir)
    mask_creator.create_masks()

    # Paths for cropping
    window_size = (256, 256)
    stride = 128
    output_folder = "/Users/asad/Documents/pcbdata/Unet_single_component/full_sub_images_opaque_new_C_only"

    # Step 2: Crop images and masks
    image_cropper = ImageCropper(image_folder, results_dir, window_size, stride, output_folder)
    image_cropper.save_cropped_images()

    # Step 3: Delete black masks
    cropped_image_folder = os.path.join(output_folder, "cropped_images")
    cropped_mask_folder = os.path.join(output_folder, "cropped_masks")
    mask_cleaner = MaskCleaner(cropped_image_folder, cropped_mask_folder)
    mask_cleaner.delete_black_masks()

    # Step 4: Synchronize images and masks
    synchronizer = ImageMaskSynchronizer(cropped_image_folder, cropped_mask_folder)
    synchronizer.synchronize_images_masks()

    # Step 5: Filter masks by non-zero content
    mask_filter = MaskFilter(cropped_mask_folder)
    mask_filter.filter_masks(threshold=0.05)


if __name__ == "__main__":
    main()

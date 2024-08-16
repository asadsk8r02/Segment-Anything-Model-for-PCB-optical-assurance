import os
import cv2
import random
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from keras import backend as K

class DataGenerator:
    def __init__(self, img_folder, mask_folder, batch_size, imsize=256):
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.batch_size = batch_size
        self.imsize = imsize
        self.img_files = sorted(os.listdir(img_folder))
        self.mask_files = sorted(os.listdir(mask_folder))
        self.zipped_files = list(zip(self.img_files, self.mask_files))
        random.shuffle(self.zipped_files)
        self.img_files, self.mask_files = zip(*self.zipped_files)
        self.c = 0
        self.total = len(self.img_files)

    def __iter__(self):
        return self

    def __next__(self):
        img = np.zeros((self.batch_size, self.imsize, self.imsize, 3)).astype('float')
        mask = np.zeros((self.batch_size, self.imsize, self.imsize, 1)).astype('float')

        for i in range(self.batch_size):
            if self.c == self.total:
                self.c = 0
            train_img = cv2.imread(os.path.join(self.img_folder, self.img_files[self.c])) / 255.0
            train_img = cv2.resize(train_img, (self.imsize, self.imsize))

            train_mask = cv2.imread(os.path.join(self.mask_folder, self.mask_files[self.c]), cv2.IMREAD_GRAYSCALE)
            train_mask = cv2.resize(train_mask, (self.imsize, self.imsize))
            train_mask = np.expand_dims(train_mask, axis=-1) / 255.0

            img[i] = train_img
            mask[i] = train_mask
            self.c += 1

        return img, mask


class ModelEvaluator:
    def __init__(self, model_path, data_generator, batch_size):
        self.model = tf.keras.models.load_model(model_path)
        self.data_generator = data_generator
        self.batch_size = batch_size

    def evaluate(self):
        test_steps = math.ceil(len(self.data_generator.img_files) / self.batch_size)
        results = self.model.evaluate(self.data_generator, steps=test_steps)
        print("Test Loss:", results[0])
        print("Test Accuracy:", results[1])

    def predict_and_display(self):
        for images, masks in self.data_generator:
            sample_image, sample_mask = images[2], masks[2]
            break

        predicted_mask = self.model.predict(sample_image[np.newaxis, ...])
        predicted_mask = (predicted_mask > 0.4).astype(np.float32)
        self.display([sample_image, sample_mask, predicted_mask[0]])

    @staticmethod
    def display(display_list):
        plt.figure(figsize=(15, 15))
        title = ['Input Image', 'True Mask', 'Predicted Mask']
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()


def main():
    img_folder = "/Users/asad/Documents/pcbdata/Unet_single_component/full_sub_images_opaque_new_BTN_only/data/test/images"
    mask_folder = "/Users/asad/Documents/pcbdata/Unet_single_component/full_sub_images_opaque_new_BTN_only/data/test/masks"
    model_path = "/Users/asad/Documents/PCB_files/models/unet_model_BTN.h5"
    batch_size = 4

    # Create data generator
    data_generator = DataGenerator(img_folder, mask_folder, batch_size)

    # Evaluate model
    evaluator = ModelEvaluator(model_path, data_generator, batch_size)
    evaluator.evaluate()

    # Predict and display results
    evaluator.predict_and_display()


if __name__ == "__main__":
    main()

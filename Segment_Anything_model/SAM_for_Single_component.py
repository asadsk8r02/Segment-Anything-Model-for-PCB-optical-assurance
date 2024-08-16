import os
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import SamProcessor, SamModel, SamConfig
import monai
from tqdm import tqdm


class ImageLoader:
    def __init__(self, image_folder, mask_folder, target_size=(256, 256)):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.target_size = target_size
        self.images = []
        self.masks = []

    def load_data(self):
        image_filenames = [filename for filename in os.listdir(self.image_folder) if filename.endswith(('.jpg', '.jpeg', '.png'))]
        for image_filename in image_filenames:
            image_path = os.path.join(self.image_folder, image_filename)
            mask_path = os.path.join(self.mask_folder, image_filename.replace("_image.png", "_mask.png"))
            self.images.append(self.load_image(image_path))
            self.masks.append(self.load_mask(mask_path))
        return np.array(self.images), np.array(self.masks)

    def load_image(self, image_path):
        with Image.open(image_path) as image:
            image = image.resize(self.target_size, Image.BILINEAR)
            return np.array(image)

    def load_mask(self, mask_path):
        with Image.open(mask_path) as mask:
            mask = mask.resize(self.target_size, Image.BILINEAR)
            return (np.array(mask) / 255.0).astype(np.uint8)


class BoundingBox:
    @staticmethod
    def get_bounding_box(ground_truth_map):
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        return [x_min, y_min, x_max, y_max]

    @staticmethod
    def visualize_bounding_boxes(image, ground_truth_mask, bbox):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(ground_truth_mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
        plt.tight_layout()
        plt.show()


class SAMDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])
        prompt = BoundingBox.get_bounding_box(ground_truth_mask)
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = ground_truth_mask
        return inputs


class SAMTrainer:
    def __init__(self, model, processor, dataset, batch_size=2, learning_rate=1e-5, weight_decay=0):
        self.model = model
        self.processor = processor
        self.dataset = SAMDataset(dataset, processor)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        self.optimizer = torch.optim.Adam(self.model.mask_decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.model.to(self.device)

    def train(self, num_epochs=100):
        self.model.train()
        for epoch in range(num_epochs):
            epoch_losses, epoch_accuracies, epoch_precisions, epoch_recalls, epoch_f1_scores = [], [], [], [], []
            for batch in tqdm(self.dataloader):
                loss, accuracy, precision, recall, f1 = self.train_batch(batch)
                epoch_losses.append(loss.item())
                epoch_accuracies.append(accuracy.item())
                epoch_precisions.append(precision)
                epoch_recalls.append(recall)
                epoch_f1_scores.append(f1)
            self.log_epoch_results(epoch, epoch_losses, epoch_accuracies, epoch_precisions, epoch_recalls, epoch_f1_scores)

    def train_batch(self, batch):
        pixel_values = batch["pixel_values"].to(self.device)
        ground_truth_masks = batch["ground_truth_mask"].to(self.device)
        input_boxes = batch["input_boxes"].float().to(self.device)

        outputs = self.model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
        predicted_masks = outputs.pred_masks.squeeze(1)
        loss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1).float())

        predicted_masks_binary = (predicted_masks > 0.5).float()
        accuracy = (predicted_masks_binary == ground_truth_masks).float().mean()

        ground_truth_masks_np = ground_truth_masks.cpu().numpy().flatten()
        predicted_masks_binary_np = predicted_masks_binary.cpu().numpy().flatten()
        precision = precision_score(ground_truth_masks_np, predicted_masks_binary_np, zero_division=0)
        recall = recall_score(ground_truth_masks_np, predicted_masks_binary_np, zero_division=0)
        f1 = f1_score(ground_truth_masks_np, predicted_masks_binary_np, zero_division=0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, accuracy, precision, recall, f1

    def log_epoch_results(self, epoch, losses, accuracies, precisions, recalls, f1_scores):
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {np.mean(losses)}')
        print(f'Mean accuracy: {np.mean(accuracies)}')
        print(f'Mean precision: {np.mean(precisions)}')
        print(f'Mean recall: {np.mean(recalls)}')
        print(f'Mean F1 score: {np.mean(f1_scores)}')

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)


class SAMInference:
    def __init__(self, model, processor, dataset, device):
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.device = device

    def evaluate(self, idx):
        test_image, ground_truth_mask = self.load_sample(idx)
        inputs = self.prepare_inputs(test_image, ground_truth_mask)
        outputs = self.run_inference(inputs)
        medsam_seg_prob, medsam_seg = self.process_output(outputs)
        self.plot_results(test_image, ground_truth_mask, medsam_seg, medsam_seg_prob)
        return medsam_seg, ground_truth_mask

    def load_sample(self, idx):
        test_image = self.dataset[idx]["image"]
        ground_truth_mask = np.array(self.dataset[idx]["label"])
        return test_image, ground_truth_mask

    def prepare_inputs(self, test_image, ground_truth_mask):
        prompt = BoundingBox.get_bounding_box(ground_truth_mask)
        inputs = self.processor(test_image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.float().to(self.device) for k, v in inputs.items()}
        return inputs

    def run_inference(self, inputs):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        return outputs

    def process_output(self, outputs):
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.8).astype(np.uint8)
        return medsam_seg_prob, medsam_seg

    def plot_results(self, test_image, ground_truth_mask, medsam_seg, medsam_seg_prob):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(np.array(test_image), cmap='gray')
        axes[0].set_title("Image")
        axes[1].imshow(ground_truth_mask, cmap='gray')
        axes[1].set_title("True Mask")
        axes[2].imshow(medsam_seg, cmap='gray')
        axes[2].set_title("Predicted Mask")
        axes[3].imshow(medsam_seg_prob)
        axes[3].set_title("Probability Map")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()


def main():
    image_folder = "/Users/asad/Documents/pcbdata/Unet_single_component/SAM_BTN/images"
    mask_folder = "/Users/asad/Documents/pcbdata/Unet_single_component/SAM_BTN/masks"

    image_loader = ImageLoader(image_folder, mask_folder)
    cropped_images, cropped_masks = image_loader.load_data()

    valid_indices = [i for i, mask in enumerate(cropped_masks) if mask.max() != 0]
    filtered_images = cropped_images[valid_indices]
    filtered_masks = cropped_masks[valid_indices]

    dataset_dict = {
        "image": [Image.fromarray(img) for img in filtered_images],
        "label": [Image.fromarray(mask) for mask in filtered_masks],
    }

    dataset = Dataset.from_dict(dataset_dict)

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")

    trainer = SAMTrainer(model, processor, dataset)
    trainer.train(num_epochs=100)
    trainer.save_model("/Users/asad/Documents/pcbdata/Unet_single_component/SAM_BTN/models/model_checkpoint.pth")

    # Inference
    inference = SAMInference(model, processor, dataset, trainer.device)
    inference.evaluate(random.randint(0, filtered_images.shape[0]-1))

if __name__ == "__main__":
    main()

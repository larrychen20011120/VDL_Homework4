import os
from PIL import Image
import random
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A

def random_augment(image1, image2):

    H = image1.size[0]
    W = image2.size[1]
    patch_size = 128
    ind_H = random.randint(0, H - patch_size)
    ind_W = random.randint(0, W - patch_size)

    image1 = image1.crop((ind_W, ind_H, ind_W + patch_size, ind_H + patch_size))
    image2 = image2.crop((ind_W, ind_H, ind_W + patch_size, ind_H + patch_size))

    if random.random() > 0.5:
        degree = random.sample([0, 90, 180, 270], k=1)[0]
        image1 = transforms.functional.rotate(image1, angle=degree)
        image2 = transforms.functional.rotate(image2, angle=degree)

    else:
        image1 = transforms.functional.hflip(image1)
        image2 = transforms.functional.hflip(image2)
        degree = random.sample([0, 90, 180, 270], k=1)[0]
        image1 = transforms.functional.rotate(image1, angle=degree)
        image2 = transforms.functional.rotate(image2, angle=degree)

    return image1, image2



class HW4Dataset(Dataset):
    def __init__(self, rain_ids, snow_ids, is_train=True):
        
        super().__init__()
        
        self.is_train = is_train

        self.id2name = dict()
        self.labels  = [
            0 if i < len(snow_ids) else 1
            for i in range(len(snow_ids)+len(rain_ids))
        ]

        for id in range(len(snow_ids)+len(rain_ids)):
            if id < len(snow_ids):
                self.id2name[id] = f"snow-{snow_ids[id]}"
            else:
                self.id2name[id] = f"rain-{rain_ids[id-len(snow_ids)]}"

        
        snow_image_paths = [
            os.path.join("hw4_release_dataset", "train", "degraded", f"snow-{snow_id}.png")
            for snow_id in snow_ids
        ]
        rain_image_paths = [
            os.path.join("hw4_release_dataset", "train", "degraded", f"rain-{rain_id}.png")
            for rain_id in rain_ids
        ]
        snow_clean_paths = [
            os.path.join("hw4_release_dataset", "train", "clean", f"snow_clean-{snow_id}.png")
            for snow_id in snow_ids
        ]
        rain_clean_paths = [
            os.path.join("hw4_release_dataset", "train", "clean", f"rain_clean-{rain_id}.png")
            for rain_id in rain_ids
        ]

        self.degraded_image_paths = snow_image_paths + rain_image_paths
        self.clean_image_paths = snow_clean_paths + rain_clean_paths

    def __getitem__(self, idx):
        degraded_path = self.degraded_image_paths[idx]
        clean_path = self.clean_image_paths[idx]
        label = self.labels[idx]
        

        if self.is_train:

            degrad_img = Image.open(degraded_path).convert('RGB')
            clean_img =  Image.open(clean_path).convert('RGB')

            degrad_img, clean_img = random_augment(degrad_img, clean_img)


        else:
            degrad_img = Image.open(degraded_path).convert('RGB')
            clean_img =  Image.open(clean_path).convert('RGB')

        degrad_img = transforms.ToTensor()(degrad_img)
        clean_img = transforms.ToTensor()(clean_img)
        label = torch.tensor(label, dtype=torch.long)

        return label, degrad_img, clean_img
    
    def __len__(self):
        return len(self.degraded_image_paths)
    

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == "__main__":

    snow_ids = [1, 2, 3, 4, 5]  # Example IDs for snow images
    rain_ids = [1, 2, 3, 4, 5]  # Example IDs for rain images

    # Initialize the dataset and dataloader
    dataset = HW4Dataset(snow_ids=snow_ids, rain_ids=rain_ids, is_train=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Example batch size of 4
    
    # Get a single batch from the dataloader
    batch = next(iter(dataloader))
    
    # Get image details
    batch_names, degraded_images, clean_images = batch

    # Plot the images
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))

    for i in range(4):
        # Original degraded image
        axes[i, 0].imshow(np.transpose(degraded_images[i], (1, 2, 0)))  # Convert from CHW to HWC
        axes[i, 0].axis('off')

        # Clean image
        axes[i, 1].imshow(np.transpose(clean_images[i], (1, 2, 0)))  # Convert from CHW to HWC
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig("example_augmented_images.png")
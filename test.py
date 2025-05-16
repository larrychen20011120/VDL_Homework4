import argparse
from tqdm import tqdm
import numpy as np

import torch
import os
import torch.nn as nn 
from torchvision import transforms

from net.model import PromptIR

import lightning.pytorch as pl
from PIL import Image

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--ckpt_name', type=str)

    args = parser.parse_args()

    net = PromptIRModel.load_from_checkpoint(args.ckpt_name)
    net = net.to(args.device)
    net.eval()

    entry = os.path.join("hw4_release_dataset", "test", "degraded")
    images_dict = {}

    for filename in tqdm(os.listdir(entry)):

        image_path = os.path.join(entry, filename)

        degrad_img = Image.open(image_path).convert('RGB')

        with torch.no_grad():
            
            degrad_img = transforms.ToTensor()(degrad_img)
            degrad_img = degrad_img.unsqueeze(0)
            degrad_img = degrad_img.to(args.device)

            restored_img = net(degrad_img)
            restored_img =  restored_img.squeeze(0)
            restored_img *= 255
            restored_img = restored_img.cpu().numpy().astype(np.uint8)

        images_dict[filename] = restored_img
    
    np.savez('pred.npz', **images_dict)
        
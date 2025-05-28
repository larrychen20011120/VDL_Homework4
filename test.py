import argparse
from tqdm import tqdm
import numpy as np

import torch
import os
import torch.nn as nn 
from torchvision import transforms
import torchvision.transforms.functional as TF

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
    
class PromptIRModelUEM(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.net = PromptIR(decoder=True)
        
        self.restored_loss_fn  = nn.L1Loss()
        # self.ssim_loss_fn = SSIM(data_range=1.0, size_average=True, channel=3)
        self.log_sigma_l1 = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.log_sigma_ssim = torch.nn.Parameter(torch.tensor(-2.0), requires_grad=True)

    def forward(self,x):
        return self.net(x)

class PromptIRModelSSIM(pl.LightningModule):
    def __init__(self):
        
        super().__init__()

        self.net = PromptIR(decoder=True)
        # self.alpha = torch.nn.Parameter(torch.tensor(-3.0), requires_grad=True)

    def forward(self,x):
        return self.net(x)
    

def predict(net, img_patch, tta=False):
    """
    对单个 patch（1×C×128×128）做 4 种翻转 TTA，返回平均后的预测（1×C×128×128）。
    """
    flips = [None, 'h', 'v', 'hv'] if tta else [None]
    preds = []
    for f in flips:
        # 1）做翻转
        if f is None:
            x = img_patch
        elif f == 'h':
            x = TF.hflip(img_patch)
        elif f == 'v':
            x = TF.vflip(img_patch)
        else:  # 'hv'
            x = TF.hflip(TF.vflip(img_patch))

        # 2）前向
        out = net(x)
        if isinstance(out, tuple):  # 如果模型返回 (restored, other)
            out = out[0]

        # 3）反向翻转回原始方向
        if f is None:
            y = out
        elif f == 'h':
            y = TF.hflip(out)
        elif f == 'v':
            y = TF.vflip(out)
        else:
            y = TF.vflip(TF.hflip(out))

        preds.append(y)

    # 4）对 4 次预测取平均
    return torch.stack(preds, dim=0).mean(dim=0, keepdim=True)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--method', type=int, default=0)
    parser.add_argument('--ckpt_name', type=str)
    parser.add_argument('--tta', type=int, default=0)

    args = parser.parse_args()

    if args.method == 0:
        net = PromptIRModel.load_from_checkpoint(args.ckpt_name)
        
    elif args.method == 1:
        net = PromptIRModelUEM.load_from_checkpoint(args.ckpt_name)
        
    elif args.method == 2:
        net = PromptIRModelSSIM.load_from_checkpoint(args.ckpt_name)

    net = net.to(args.device)
    net.eval()

    entry = os.path.join("hw4_release_dataset", "test", "degraded")
    images_dict = {}

    for fn in tqdm(os.listdir(entry)):
        # 读取 degraded image
        img = Image.open(os.path.join(entry, fn)).convert('RGB')
        img = transforms.ToTensor()(img).unsqueeze(0).to(args.device)  # 1×3×256×256

        with torch.no_grad():
            # 切成 4 个 patch
            patches = [
                img[:, :,   :128,   :128],  # top-left
                img[:, :,   :128, 128:  ],  # top-right
                img[:, :, 128:  ,   :128],  # bottom-left
                img[:, :, 128:  , 128:  ],  # bottom-right
            ]

            # 对每个 patch 做 TTA
            pred_patches = [predict(net, p, args.tta) for p in patches]

            # 拼回 full image
            restored = torch.zeros_like(img)
            restored[:, :,   :128,   :128] = pred_patches[0]
            restored[:, :,   :128, 128:  ] = pred_patches[1]
            restored[:, :, 128:  ,   :128] = pred_patches[2]
            restored[:, :, 128:  , 128:  ] = pred_patches[3]

            # 转 numpy、放大到 [0,255]
            out_np = (restored.squeeze(0).clamp(0,1) * 255).cpu().numpy().astype(np.uint8)

        images_dict[fn] = out_np

    # 保存为 npz
    np.savez('pred.npz', **images_dict)
        
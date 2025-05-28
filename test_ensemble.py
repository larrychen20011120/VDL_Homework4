import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os

from net.model import PromptIR
import lightning.pytorch as pl

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

def tta_predict(net, img_patch):
    """对单个 patch 做 4 向翻转 TTA，返回平均预测"""
    flips = [None, 'h', 'v', 'hv']
    preds = []
    for f in flips:
        # 翻转
        if   f is None: x = img_patch
        elif f == 'h':  x = TF.hflip(img_patch)
        elif f == 'v':  x = TF.vflip(img_patch)
        else:           x = TF.hflip(TF.vflip(img_patch))
        # 前向
        out = net(x)
        if isinstance(out, tuple):
            out = out[0]
        # 反向翻转
        if   f is None: y = out
        elif f == 'h':  y = TF.hflip(out)
        elif f == 'v':  y = TF.vflip(out)
        else:           y = TF.vflip(TF.hflip(out))
        preds.append(y)
    return torch.stack(preds, dim=0).mean(dim=0)

def sliding_window_patches(img, patch_size, stride):
    """生成所有滑动窗口的 (x0,y0) 左上角坐标列表"""
    _, _, H, W = img.shape
    xs = list(range(0, W - patch_size + 1, stride))
    ys = list(range(0, H - patch_size + 1, stride))
    # 确保右/下边缘能覆盖到
    if xs[-1] + patch_size < W:
        xs.append(W - patch_size)
    if ys[-1] + patch_size < H:
        ys.append(H - patch_size)
    coords = [(y, x) for y in ys for x in xs]
    return coords

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',    type=str, default="cuda")
    parser.add_argument('--method',    type=int, default=0)
    parser.add_argument('--ckpt_name', type=str, required=True)
    parser.add_argument('--patch',     type=int, default=128)
    parser.add_argument('--stride',    type=int, default=64)
    args = parser.parse_args()

    # 加载模型
    if args.method == 0:
        net = PromptIRModel.load_from_checkpoint(args.ckpt_name)
    elif args.method == 1:
        net = PromptIRModelUEM.load_from_checkpoint(args.ckpt_name)
    else:
        net = PromptIRModelSSIM.load_from_checkpoint(args.ckpt_name)
    net = net.to(args.device).eval()

    transform = transforms.ToTensor()
    entry = "hw4_release_dataset/test/degraded"

    images_dict = {}

    for fn in tqdm(os.listdir(entry)):
        # 读图
        img_pil = Image.open(os.path.join(entry, fn)).convert('RGB')
        img = transform(img_pil).unsqueeze(0).to(args.device)  # 1×3×256×256

        B, C, H, W = img.shape
        patch, stride = args.patch, args.stride

        # 准备累加和权重图
        recon_sum = torch.zeros_like(img)
        weight   = torch.zeros_like(img)

        # 遍历所有 patch
        coords = sliding_window_patches(img, patch, stride)
        with torch.no_grad():
            for (y0, x0) in coords:
                patch_in = img[:, :, y0:y0+patch, x0:x0+patch]
                patch_out = tta_predict(net, patch_in)  # 1×3×patch×patch

                # 累加
                recon_sum[:, :, y0:y0+patch, x0:x0+patch] += patch_out
                weight  [:, :, y0:y0+patch, x0:x0+patch] += 1

        # 除以 weight 得平均
        restored = recon_sum / weight
        out_np = (restored.squeeze(0).clamp(0,1) * 255).cpu().numpy().astype(np.uint8)

        images_dict[fn] = out_np

    # 保存为 npz
    np.savez('pred.npz', **images_dict)


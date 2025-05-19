import argparse
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# from utils.dataset_utils import PromptTrainDataset
from dataset_utils import HW4Dataset
from net.model import PromptIR, UEM
from val_utils import compute_psnr_ssim
import numpy as np

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint



class PromptIRModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        self.net = PromptIR(decoder=True)
        self.uem = UEM(256, 2)
        
        self.restored_loss_fn  = nn.L1Loss(reduction='none')
        self.opt = opt

        self.automatic_optimization = False
    
    def forward(self,x):
        return self.net(x)
    
    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        scheduler.step() 
        lr = scheduler.get_last_lr()[0]
        self.log("lr", lr)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        (label, degrad_img, clean_img) = batch

        optimizer = self.optimizers()

        restored = self.net(degrad_img)            # (B,3,H,W)

        # ❶ 像素 L1
        pix_loss = self.restored_loss_fn(restored, clean_img).mean(1)   # (B,H,W)

        # ❷ 估計 log σ² (detach restored → 按論文只對 UEM 回傳梯度，保護主網)
        log_sigma2 = self.uem(restored.detach())      # (B,K,H,W)
        sigma2     = torch.exp(log_sigma2).clamp(1e-6)

        # ❸ 依 label 換通道
        sigma2_sel = sigma2[torch.arange(restored.size(0)), label]  # (B,H,W)

        # ❹ TUR 像素 loss
        tur = pix_loss / (2.0 * sigma2_sel) + 0.5 * torch.log(sigma2_sel)
        total_loss = tur.mean() / self.opt.grad_accumulation

        self.manual_backward(total_loss)

        if (batch_idx + 1) % self.opt.grad_accumulation == 0:
            self.clip_gradients(optimizer, self.opt.clip_norm, "norm")
            optimizer.step()
            optimizer.zero_grad()

        self.log("Train/L_total", total_loss)
        for k in range(log_sigma2.size(1)):
            self.log(f"sigma2/task{k}", sigma2[:, k].mean())
            self.log("Train/Total Loss", total_loss)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        (label, degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        
        loss = self.restored_loss_fn(restored, clean_patch).mean()

        psnr, ssim, _ = compute_psnr_ssim(restored, clean_patch)

        self.log("Val/Total Loss", loss)

        self.log("Val/PSNR", psnr)
        self.log("Val/SSIM", ssim)

    
    def configure_optimizers(self):
        
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.opt.lr,
            weight_decay=self.opt.weight_decay,
        )

        warmup_epochs = self.opt.warmup_epochs
        

        if warmup_epochs != 0:

            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-4,     
                end_factor=1.0,        
                total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.opt.epochs - warmup_epochs,
                eta_min=1e-6,
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
            
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.opt.epochs,
                eta_min=1e-7,
            )
        
        return [optimizer],[scheduler]



import os

name = "PromptIR-UEM"
log_entry = os.path.join("logs", name)

def main(opt):    
    
    logger = TensorBoardLogger(save_dir = log_entry)

    ### train test split

    seed = 313551058
    val_size = int(0.2 * 1600)
    ids = list(range(1, 1601))
    fold = opt.fold
    random.seed(seed)

    random.shuffle(ids)
    train_snow_ids = ids[:fold*val_size] + ids[(fold+1)*val_size:]
    val_snow_ids = ids[fold*val_size:(fold+1)*val_size]
    
    random.shuffle(ids)
    train_rain_ids = ids[:fold*val_size] + ids[(fold+1)*val_size:]
    val_rain_ids = ids[fold*val_size:(fold+1)*val_size]

    ### train test split

    trainset = HW4Dataset(train_snow_ids, train_rain_ids, is_train=True)
    valset = HW4Dataset(val_snow_ids, val_rain_ids, is_train=False)

    print(f"training size: {len(trainset)} / val size: {len(valset)}")

    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(
        trainset, 
        batch_size=opt.batch_size, 
        pin_memory=True, 
        shuffle=True,
        num_workers=opt.num_workers
    )
    
    valloader = DataLoader(
        valset, 
        batch_size=8, 
        pin_memory=True, 
        shuffle=False,         
        num_workers=opt.num_workers
    )
    
    model = PromptIRModel(opt)
    
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        logger=logger,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
        precision=16,
    )


    trainer.fit(
        model=model, 
        train_dataloaders=trainloader, 
        val_dataloaders=valloader
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)

    parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs to train the total model.')
    parser.add_argument('--batch_size', type=int, default=3,help="Batch size to use per GPU")
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of encoder.')
    parser.add_argument('--grad_accumulation', type=int, default=6, help='Gradient accumulation steps.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay of encoder.')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='gradient clipping norm.')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='number of warmup epochs.')
    parser.add_argument('--loss_w', type=float, default=0.1, help='loss weight of classification')
    parser.add_argument('--fold', type=int, default=0, help='which fold to run.')

    # path
    parser.add_argument('--ckpt_path', type=str, default="ckpt/", help='checkpoint save path')
    parser.add_argument("--ckpt_dir",type=str,default=f"train_ckpt/{name}",help = "Name of the Directory where the checkpoint is to be saved")
    parser.add_argument("--num_gpus",type=int,default= 1,help = "Number of GPUs to use for training")

    opt = parser.parse_args()

    if not os.path.exists(log_entry):
        os.makedirs(log_entry)

    with open(os.path.join(log_entry, 'hyper_parameters.txt'), 'w') as f:
        json.dump(opt.__dict__, f, indent=4)

    
    main(opt)




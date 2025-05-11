import argparse
import json
import random

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

# from utils.dataset_utils import PromptTrainDataset
from utils.dataset_utils import HW4Dataset
from net.model import PromptIR
# from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.val_utils import compute_psnr_ssim
import numpy as np

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint



class PromptIRModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
        self.opt = opt

        self.automatic_optimization = False
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_img, clean_img) = batch

        optimizer = self.optimizers()

        restored = self.net(degrad_img)

        loss = self.loss_fn(restored,clean_img) / self.opt.grad_accumulation
        self.manual_backward(loss)

        if (batch_idx + 1) % self.opt.grad_accumulation == 0:
            self.clip_gradients(optimizer, gradient_clip_val=self.opt.clip_norm, gradient_clip_algorithm="norm")
            optimizer.step()
            optimizer.zero_grad()

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored,clean_patch)

        psnr, ssim, _ = compute_psnr_ssim(restored, clean_patch)

        self.log("val_loss", loss)
        self.log("val_psnr", psnr)
        self.log("val_ssim", ssim)

    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        self.log("lr", lr)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.opt.lr,
            weight_decay=self.opt.weight_decay,
        )
        # scheduler = CosineAnnealingLR(
        #     optimizer=optimizer,
        #     T_max=self.opt.epochs,
        #     eta_min=1e-7,
        # )
        scheduler = MultiStepLR(
            optimizer=optimizer,
            milestones=[15, 45, 75, 90],
            gamma=0.1,
        )
        return [optimizer],[scheduler]






def main(opt):
    
    logger = TensorBoardLogger(save_dir = "logs/")

    ### train test split

    seed = 313551058
    val_size = int(0.2 * 1600)
    ids = list(range(1, 1601))
    random.seed(seed)

    random.shuffle(ids)
    train_snow_ids = ids[val_size:]
    val_snow_ids = ids[:val_size]
    
    random.shuffle(ids)
    train_rain_ids = ids[val_size:]
    val_rain_ids = ids[:val_size]

    ### train test split

    trainset = HW4Dataset(train_snow_ids, train_rain_ids, is_train=True)
    valset = HW4Dataset(val_snow_ids, val_rain_ids, is_train=False)

    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(
        trainset, 
        batch_size=opt.batch_size, 
        pin_memory=True, 
        shuffle=True,
        drop_last=True, 
        num_workers=opt.num_workers
    )
    
    valloader = DataLoader(
        valset, 
        batch_size=opt.batch_size, 
        pin_memory=True, 
        shuffle=False,         
        drop_last=True, 
        num_workers=opt.num_workers
    )
    
    model = PromptIRModel(opt)
    
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        logger=logger,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=2,
        precision=opt.precision,
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
    parser.add_argument('--batch_size', type=int,default=6,help="Batch size to use per GPU")
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of encoder.')
    parser.add_argument('--grad_accumulation', type=int, default=3, help='Gradient accumulation steps.')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay of encoder.')
    parser.add_argument('--clip_norm', type=float, default=0.5, help='gradient clipping norm.')
    parser.add_argument('--precision', type=int, default=16, help='precision of training.')
    parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

    # path
    parser.add_argument('--ckpt_path', type=str, default="ckpt/", help='checkpoint save path')
    parser.add_argument("--ckpt_dir",type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
    parser.add_argument("--num_gpus",type=int,default= 1,help = "Number of GPUs to use for training")

    opt = parser.parse_args()

    with open('hyper_parameters.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=4)

    
    main(opt)




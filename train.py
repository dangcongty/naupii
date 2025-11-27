import os
import shutil
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import Config, DensityModel, NaupiiDataset


class Trainer:
    def __init__(self):
        self._init_hyp()
        self._init_model()
        self._init_loader()

    def _init_model(self):
        self.model = DensityModel().to(self.device)
        ckpt = torch.load('ckpt/02012025.pth')
        self.model.load_state_dict(state_dict=ckpt)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                               lr = self.config.lr)
    
    def _init_hyp(self):
        self.run_name = uuid4()
        self.config = Config(
            epochs=100,
            lr=0.01,
            batch_size=16,
            data_path='datasets/',
            run_dir=f'runs/{self.run_name}'
        )
        self.device = self.config.device
        self.criteria = torch.nn.MSELoss(reduce='sum')

        os.makedirs(f'{self.config.run_dir}', exist_ok=True)
        shutil.copytree('modules/', f'{self.config.run_dir}/modules')
        shutil.copytree('scripts/', f'{self.config.run_dir}/scripts')
        print(f'##### Workspace create at: {self.config.run_dir} #####')

        self.best = np.inf
        self.scale_gt = 100

    def _init_loader(self):
        self.train_data = NaupiiDataset(image_paths='datasets/train/images', mode='train')
        self.val_data = NaupiiDataset(image_paths='datasets/val/images', mode='val')

        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.config.batch_size,
                                       shuffle=True,
                                       drop_last=True,
                                       pin_memory=True,
                                       )
        self.val_loader = DataLoader(self.val_data,
                                       batch_size=self.config.batch_size,
                                       pin_memory=True,
                                       )
    
    def train_1_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, bar_format='{l_bar:5}{bar:5}{r_bar:5}')
        train_losses = []
        print()
        for idx, (img, gt) in enumerate(pbar):
            img = img.to(self.device)
            gt = gt.to(self.device)

            outputs = self.model(img)

            loss = self.criteria(outputs, gt*self.scale_gt)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_description(f'Train: {epoch}\tLoss: {np.mean(train_losses):.3f}')

        return np.mean(train_losses)

    def val(self, epoch):
        self.model.eval()
        pbar = tqdm(self.val_loader, bar_format='{l_bar:5}{bar:5}{r_bar:5}')
        val_losses = []
        print()
        for idx, (img, gt) in enumerate(pbar):
            img = img.to(self.device)
            gt = gt.to(self.device)
            with torch.no_grad():
                outputs = self.model(img)
                loss = self.criteria(outputs/self.scale_gt, gt)
            
            val_losses.append(loss.item())
            pbar.set_description(f'Val: {epoch}\tLoss: {np.mean(val_losses):.3f}')

        return np.mean(val_losses)
    
    def save_ckpt(self, epoch, val_loss):
        if self.best < val_loss:
            ckpt_dir = f'{self.config.run_dir}/{epoch}_{val_loss:.4f}.pth'
            torch.save({"model_state_dict": self.model.state_dict()}, ckpt_dir)
            print(f"Save checkpoints => {ckpt_dir}")
            return True
        return False
    
    def train(self):
        for epoch in tqdm(range(1, self.config.epochs+1)):
            train_loss = self.train_1_epoch(epoch)
            val_loss = self.val(epoch)
            save = self.save_ckpt(epoch, val_loss)
            print(f"""
                Epoch: {epoch}
                Train loss: {train_loss:.4f}
                Val loss: {val_loss:.4f}
                is Save: {save}
            """)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    
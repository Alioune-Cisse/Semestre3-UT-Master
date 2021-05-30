import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pl_bolts.datasets import DummyDataset

class CIFARDataModule(pl.LightningDataModule):

    def setup(self):
        # transforms for images
        transform=transforms.Compose([transforms.ToTensor(), 
                                      transforms.Normalize((0.1307,), (0.3081,))])
      
        # prepare transforms standard to CIFAR
        CIFAR_train = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
        self.CIFAR_test = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)
        
        self.CIFAR_train, self.CIFAR_val = random_split(CIFAR_train, [45000, 5000])
    
        return self.CIFAR_train, self.CIFAR_val, self.CIFAR_test

    def train_dataloader(self):
        train, _ , _ = self.setup()
        return DataLoader(train, batch_size=64)

    def val_dataloader(self):
        _, val, _ = self.setup()
        return DataLoader(val, batch_size=64)

    def test_dataloader(self):
        _, _, test = self.setup()
        return DataLoader(test, batch_size=64)

    def dataset(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

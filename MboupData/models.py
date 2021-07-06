#%matplotlib inline
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets

import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import MboupData.dataset as dataset
import MboupData.transform as transform
import sys


sys.path.append('/content/Semestre3_MSDA')
path = '/content/IMAGES_ECHO_LABELLISEES/'
transforms = transform.TransformData()
data = dataset.LiverDataset(path, transform=transforms())
train_dl = data.train_dataloader(batch_size=10, shuffle=False)

hparams = {'lr':[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]}

class LiverModel(pl.LightningModule):
    
    def __init__(self, modell = models.vgg16, lr=0.0001):
        super(LiverModel, self).__init__()
     
        self.lr=lr
        
        # Model  ###############################################################################
        # Pretrained VGG16
        use_pretrained = True
        self.net = modell(pretrained=use_pretrained)
        # Change Output Size of Last FC Layer (4096 -> 1)
        self.net.classifier[6] = nn.Linear(in_features=self.net.classifier[6].in_features, out_features=4)
        # Specify The Layers for updating
        params_to_update = []
        update_params_name = ['classifier.6.weight', 'classifier.6.bias']

        for name, param in self.net.named_parameters():
            if name in update_params_name:
                param.requires_grad = True
                params_to_update.append(param)
            else:
                param.requires_grad = False
        # Set Optimizer
        self.optimizer = optim.SGD(params=params_to_update, lr=self.lr, momentum=0.9)
    
    # Method  ###############################################################################
    # Set Train Dataloader
    #@pl.data_loader
    
    """def hparams(self, hparams=hparams):
        self.hparams = hparams
        return self.hparams"""
    
    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(x,dim=1)
    
    def train_dataloader(self, train_dl=train_dl):
        
        return train_dl
    
    def test_dataloader(self):
        data = dataset.LiverDataset(path, transform=transforms())
        train_dl = data.test_dataloader(batch_size=10, shuffle=False)
        return train_dl

    def crossentropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.crossentropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.crossentropy_loss(logits, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.crossentropy_loss(logits, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #generator_optim = torch.optim.Adam(self.generator(), lr=1e-3)
        #disc_optim = torch.optim.Adam(self.discriminator(), lr=1e-3)
        return optimizer

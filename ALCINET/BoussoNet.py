import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pl_bolts.datasets import DummyDataset
#%matplotlib inline

#La forme de la classe est la suivante
class Net(nn.Module):
  def __init__(self, classes=10, input=1):
    super(Net,self).__init__()
    self.conv1 = nn.Conv2d(input,20,5,1)
    self.conv2 = nn.Conv2d(20,50,5,1)
    self.fc1 = nn.Linear(1600,500)
    self.fc2 = nn.Linear(500,classes)
  def forward(self,x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x,2)  
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x,2)
    x = x.view(-1,1600)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x,dim=1)
    
    
class NetLight(pl.LightningModule):
    def __init__(self, classes=10, input=1):
        super(NetLight,self).__init__()
        self.conv1 = nn.Conv2d(input,20,5,1)
        self.conv2 = nn.Conv2d(20,50,5,1)
        self.fc1 = nn.Linear(1600,500)
        self.fc2 = nn.Linear(500,classes)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)  
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1,1600)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)
        
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        #generator_optim = torch.optim.Adam(self.generator(), lr=1e-3)
        #disc_optim = torch.optim.Adam(self.discriminator(), lr=1e-3)
        return optimizer
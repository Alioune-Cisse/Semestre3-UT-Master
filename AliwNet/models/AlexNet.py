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


class ConvBlock(nn.Module):
    def __init__(self, input, output, kernel=1, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=input, out_channels=output, kernel_size=kernel, stride=stride, padding=padding)
        self.batch = nn.BatchNorm2d(output)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        return x
    
    
class AlexNet(nn.Module):
    def __init__(self, classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = ConvBlock(3, 96, kernel=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = ConvBlock(96, 256, kernel=5, stride=1, padding=2)
        #self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv3 = ConvBlock(256, 384, kernel=3, stride=1, padding=1)
        self.conv4 = ConvBlock(384, 384, kernel=3, stride=1, padding=1)
        self.conv5 = ConvBlock(384, 256, kernel=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc = nn.Linear(512, classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        #x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.dropout(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x
    
    
class AlexNetLight(pl.LightningModule):
    def __init__(self, classes=10):
        super(AlexNetLight, self).__init__()
        self.conv1 = ConvBlock(3, 96, kernel=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = ConvBlock(96, 256, kernel=5, stride=1, padding=2)
        #self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv3 = ConvBlock(256, 384, kernel=3, stride=1, padding=1)
        self.conv4 = ConvBlock(384, 384, kernel=3, stride=1, padding=1)
        self.conv5 = ConvBlock(384, 256, kernel=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc = nn.Linear(512, classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        #x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.dropout(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x
    
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

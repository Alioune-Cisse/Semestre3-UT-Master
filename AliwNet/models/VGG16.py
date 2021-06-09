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
  def __init__(self, input, output, kernel, padding, stride):
    super(ConvBlock, self).__init__()
    #padding = (output-1+kernel-input)//2
    self.conv = nn.Conv2d(in_channels=input, out_channels=output, 
                          kernel_size=kernel, padding=padding, stride=stride)
    self.norm = nn.BatchNorm2d(output)
    self.action = nn.ReLU()
  def forward(self, x):
    x = self.conv(x)
    x = self.norm(x)
    x = self.action(x)
    return x
    
    
class VGG16(nn.Module):
    def __init__(self, classes=10):
        super(VGG16, self).__init__()
        self.conv1 = ConvBlock(3, 64, 3, 1, 2)
        self.conv2 = ConvBlock(64, 64, 3, 1, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv3 = ConvBlock(64, 128, 3, 1, 2)
        self.conv4 = ConvBlock(128, 128, 3, 1, 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv5 = ConvBlock(128, 256, 3, 1, 2)
        self.conv6 = ConvBlock(256, 256, 3, 1, 2)
        self.conv7 = ConvBlock(256, 256, 3, 1, 2)   
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv8 = ConvBlock(256, 512, 3, 1, 2)
        self.conv9 = ConvBlock(512, 512, 3, 1, 2)
        self.conv10 = ConvBlock(512, 512, 3, 1, 2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv11 = ConvBlock(512, 512, 3, 1, 2)
        self.conv12 = ConvBlock(512, 512, 3, 1, 2)
        self.conv13 = ConvBlock(512, 512, 3, 1, 2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
    
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
    
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
    
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
    
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)
        
        
class VGG16Light(pl.LightningModule):
    def __init__(self, classes=10):
        super(VGG16Light, self).__init__()
        self.conv1 = ConvBlock(3, 64, 3, 1, 2)
        self.conv2 = ConvBlock(64, 64, 3, 1, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv3 = ConvBlock(64, 128, 3, 1, 2)
        self.conv4 = ConvBlock(128, 128, 3, 1, 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv5 = ConvBlock(128, 256, 3, 1, 2)
        self.conv6 = ConvBlock(256, 256, 3, 1, 2)
        self.conv7 = ConvBlock(256, 256, 3, 1, 2)   
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv8 = ConvBlock(256, 512, 3, 1, 2)
        self.conv9 = ConvBlock(512, 512, 3, 1, 2)
        self.conv10 = ConvBlock(512, 512, 3, 1, 2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv11 = ConvBlock(512, 512, 3, 1, 2)
        self.conv12 = ConvBlock(512, 512, 3, 1, 2)
        self.conv13 = ConvBlock(512, 512, 3, 1, 2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
    
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
    
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
    
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
    
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
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

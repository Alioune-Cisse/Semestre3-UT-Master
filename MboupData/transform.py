#%matplotlib inline
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets



class TransformData():
    def __init__(self, RHF=0.5, RVF=0.5, RR=45, RRC=96):
        self.data_tr = transforms.Compose([
            transforms.RandomHorizontalFlip(p=RHF),
            transforms.RandomVerticalFlip(p=RVF),
            transforms.RandomRotation(RR),
            transforms.RandomResizedCrop(RRC,scale=(0.8,1.0),ratio=(1.0,1.0)),
            transforms.ToTensor()])
    def __call__(self):
        return self.data_tr
#%matplotlib inline
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets



class LiverDataset():
    def __init__(self, path, transform=None):
        self.dataset = datasets.ImageFolder(path, transform=transform)

    def __len__(self):
      return len(self.dataset)

    def setup(self):
        self.train, self.test = train_test_split(self.dataset, train_size = 100)
        return self.train, self.test

    def train_dataloader(self, batch_size=64, shuffle=True):
        train, _ = self.setup()
        return DataLoader(train, batch_size=batch_size, shuffle=shuffle)

    def test_dataloader(self, batch_size=64, shuffle=True):
        _, test = self.setup()
        return DataLoader(test, batch_size=batch_size, shuffle=shuffle)

    def dataset(self):
        return self.train_dataloader(),  self.test_dataloader()

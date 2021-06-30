#%matplotlib inline
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets
from torchvision import utils
import matplotlib.pyplot as plt


class ShowImage():
    def __init__(self, img):
        plt.figure(figsize=(10, 6))
        # Transformer le tenseur pytorch en numpy array
        self.img_np = img.numpy()
        # Changer le format des dimensions en H*W*C
        self.img_np_tr = np.transpose(self.img_np,(1,2,0))
        print(self.img_np_tr.shape)

    def show(self):
        plt.imshow(self.img_np_tr, interpolation='nearest')
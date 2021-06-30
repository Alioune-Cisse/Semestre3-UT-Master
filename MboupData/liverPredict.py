import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import MboupData.plot as pt
from torchvision import utils
  #%matplotlib inline

class Predict():
    def __init__(self, test_dl, model):
      self.images, self.labels = next(iter(test_dl))
      self.model = model
      with torch.no_grad():
        if torch.cuda.is_available()==True:
          self.images = self.images.type(torch.float).to(torch.device("cuda:0"))
          self.logps = self.model(self.images.type(torch.float))

    # Output of the network are log-probabilities, need to take exponential for probabilities
    def afficher(self, index):
      ps = torch.exp(self.logps)
      print(ps)
      img = self.images[index].detach().cpu()
      a = torch.argmax(ps[index])
      idx = np.unravel_index(a.detach().cpu().numpy(), ps[index].shape)
      print(idx[0], self.labels[index].item())
      x_grid = utils.make_grid(img, nrow=2, padding = 4)
      pt.ShowImage(x_grid).show()
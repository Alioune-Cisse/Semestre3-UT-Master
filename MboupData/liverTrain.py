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
  #%matplotlib inline

class TrainModel():
    def __init__(self, model):
      self.model = model

    def calculate_accuracy(self, top_class, labels):
      if top_class.shape != labels.shape:
        labels.view(*top_class.shape)
      equals = top_class == labels
      accuracy = torch.mean(equals.type(torch.FloatTensor)).to(torch.device("cuda:0"))
      return accuracy.item()

    #TrainBatch
    def train_batch(self, im, labe, loss_fn, optimizer=None):
      if torch.cuda.is_available()==True:
        im=im.type(torch.float).to(torch.device("cuda:0"))
        labe=labe.to(torch.device("cuda:0"))

      labe_pred = self.model(im)
      loss = loss_fn(labe_pred, labe.long())
      top_prob, top_class = torch.exp(labe_pred).topk(1, dim=1)
      batch_accuracy = self.calculate_accuracy(top_class, labe)
      if optimizer is not None :
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
      return loss.item(), batch_accuracy

    #TrainEpoch
    def train_epoch(self,  train_dl, test_dl, optimizer, epoch, loss_fn):
      running_loss = 0.
      test_loss = 0.
      train_accuracy = 0.
      val_accuracy = 0.
      self.model.train()
      for im,lab in train_dl:
        optimizer.zero_grad()
        x,y = self.train_batch(im, lab, loss_fn, optimizer)
        running_loss += x
        train_accuracy += y

      with torch.no_grad():
        self.model.eval()
        for im_test, lab_test in test_dl:
          x,y = self.train_batch(im_test, lab_test, loss_fn)
          test_loss += x
          val_accuracy += y
        self.model.train()

      return running_loss/len(train_dl), train_accuracy/len(train_dl), test_loss/len(train_dl), val_accuracy/len(train_dl)


    #Visualiser Tensorboard
    def visualiser_tensorboard(self, train_losses, test_losses, train_accuracies, val_accuracies):
        writer = SummaryWriter()
        for n_iter in range(len(train_losses)):
          writer.add_scalar('Loss/Train', np.array(train_losses).squeeze()[n_iter], n_iter)
          writer.add_scalar('Accuracy/Train', np.array(train_accuracies).squeeze()[n_iter], n_iter)
          writer.add_scalar('Loss/Test', np.array(test_losses).squeeze()[n_iter], n_iter)
          writer.add_scalar('Accuracy/Test', np.array(val_accuracies).squeeze()[n_iter], n_iter)
        writer.close()  


    #Train_model
    def train_model(self, epochs, optimizer, train_dl, test_dl, loss_fn):
      train_losses = []
      test_losses = []
      train_accuracies = []
      val_accuracies = []
      for epoch in range(epochs):
        x, y, z, t = self.train_epoch(train_dl, test_dl, optimizer, epoch, loss_fn)
        train_losses.append(x)
        test_losses.append(y)
        train_accuracies.append(z)
        val_accuracies.append(t)
        print(f"""Epoch : {epoch}/{epochs}, 
                  Training_loss : {train_losses[-1]},
                  Train_accuracy : {train_accuracies[-1]},
                  Test_loss : {test_losses[-1]},
                  Test_accuracy : {val_accuracies[-1]}""")
      self.visualiser_tensorboard(train_losses, test_losses, train_accuracies, val_accuracies)
      return train_losses, test_losses, train_accuracies, val_accuracies

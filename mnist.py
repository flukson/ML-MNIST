#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Solution based on following tutorials:
# https://www.aiworkbox.com/lessons/load-mnist-dataset-from-pytorch-torchvision
# https://www.codementor.io/@dejanbatanjac/pytorch-the-missing-manual-on-loading-mnist-dataset-wjeh5top7
# https://nextjournal.com/gkoehler/pytorch-mnist

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

data_subdirectory = "./data/"

bs = 512
learning_rate = 0.01
momentum = 0.5
n_epochs = 3

t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0), std=(1))])

class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x)

if __name__ == '__main__':

  mnist_trainset = datasets.MNIST(root=data_subdirectory, train=True, download=True, transform=t)
  mnist_testset = datasets.MNIST(root=data_subdirectory, train=False, download=True, transform=t)

  train_loader = DataLoader(mnist_trainset, batch_size=bs, drop_last=True, shuffle=True)
  test_loader = DataLoader(mnist_testset, batch_size=bs, drop_last=True, shuffle=True)

  # Initialize the network and the optimizer:
  network = NeuralNetwork()
  optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

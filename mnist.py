#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Solution based on following tutorials:
# https://www.aiworkbox.com/lessons/load-mnist-dataset-from-pytorch-torchvision
# https://www.codementor.io/@dejanbatanjac/pytorch-the-missing-manual-on-loading-mnist-dataset-wjeh5top7
# https://nextjournal.com/gkoehler/pytorch-mnist

import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from neural_network import NeuralNetwork
from neural_network_helper import *

data_subdirectory = "./data/"

batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5

# An epoch is a measure of the number of times all of the training vectors
# are used once to update the weights. For batch training all of the training
# samples pass through the learning algorithm simultaneously in one epoch before
# weights are updated.
n_epochs = 2

t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0), std=(1))])

if __name__ == '__main__':

  # Download data:
  mnist_trainset = datasets.MNIST(root=data_subdirectory, train=True, download=True, transform=t)
  mnist_testset = datasets.MNIST(root=data_subdirectory, train=False, download=True, transform=t)

  # Load data:
  train_loader = DataLoader(mnist_trainset, batch_size=batch_size_train, drop_last=True, shuffle=True)
  test_loader = DataLoader(mnist_testset, batch_size=batch_size_test, drop_last=True, shuffle=True)

  # Initialize the network and the optimizer:
  network = NeuralNetwork()
  optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

  train_losses = []
  train_counter = []
  test_losses = []
  test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

  # Run the training:
  loss = test(network, test_loader) # this one is to evaluate model with randomly initialized parameters
  test_losses.append(loss)
  for epoch in range(1, n_epochs + 1):
    loss, counter = train(network, epoch, train_loader, optimizer)
    train_losses.append(loss)
    train_counter.append(counter)
    loss = test(network, test_loader)
    test_losses.append(loss)

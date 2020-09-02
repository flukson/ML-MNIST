#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Solution based on following tutorials:
# https://www.aiworkbox.com/lessons/load-mnist-dataset-from-pytorch-torchvision
# https://www.codementor.io/@dejanbatanjac/pytorch-the-missing-manual-on-loading-mnist-dataset-wjeh5top7
# https://nextjournal.com/gkoehler/pytorch-mnist

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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

log_interval = 10

t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0), std=(1))])

class NeuralNetwork(nn.Module):

  def __init__(self):
    '''
    This function contains definitions of layers in the neural network
    '''

    super(NeuralNetwork, self).__init__()

    # Convolutional layers are used for extraction of features.
    # Nice movie about CNN networks: https://www.youtube.com/watch?v=RANnxwUGAks

    # First 2D convolutional layer, taking in 1 input channel (image),
    # outputting 10 convolutional features, with a square kernel size of 5
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

    # Second 2D convolutional layer, taking in 10 input channels,
    # outputting 20 convolutional features, with a square kernel size of 5
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

    # Dropout layer to prevent overfitting
    self.conv2_drop = nn.Dropout2d()

    # First fully connected layer
    self.fc1 = nn.Linear(320, 50)

    # Second fully connected layer that outputs 10 labels
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    '''
    This function defines how data pass through subsequent layers of the neural network
    Args:
        x: data
    '''

    # Maximum pooling, or max pooling, is a pooling operation that calculates
    # the maximum, or largest, value in each patch of each feature map.
    # Here size of pool of square window is 2.
    # Max pooling is used to reduce the amount of information.

    # relu - use the rectified-linear (ReLU) activation function (a piecewise linear function
    # that will output the input directly if it is positive, otherwise, it will output zero)
    x = F.relu(F.max_pool2d(self.conv1(x), 2))

    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

    # https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch
    # Reshape output from convolutional layer to have 320 columns
    # (number of inputs of the 1st fully connected layer). -1 means that the number of rows
    # will be adjusted automatically.
    # Output from the CNN network must be flattened before sending to classifying neural network.
    x = x.view(-1, 320)

    # Apply dropout
    x = F.dropout(x, training=self.training)

    # Use the rectified-linear activation function over x
    x = F.relu(self.fc1(x))

    # Apply softmax to x
    # https://en.wikipedia.org/wiki/Softmax_function
    # Softmax function - generalization of the logistic regression function to multiple dimensions
    return F.log_softmax(self.fc2(x))

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

  def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      output = network(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
        loss = loss.item()
        counter = batch_idx*64 + (epoch-1)*len(train_loader.dataset)
        torch.save(network.state_dict(), './results/model.pth')
        torch.save(optimizer.state_dict(), './results/optimizer.pth')
    return loss, counter

  def test():
    network.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
        output = network(data)
        loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return loss

  # Run the training:
  loss = test() # this one is to evaluate model with randomly initialized parameters
  test_losses.append(loss)
  for epoch in range(1, n_epochs + 1):
    loss, counter = train(epoch)
    train_losses.append(loss)
    train_counter.append(counter)
    loss = test()
    test_losses.append(loss)

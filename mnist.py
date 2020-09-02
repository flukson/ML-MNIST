#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Solution based on following tutorials:
# [1] https://www.aiworkbox.com/lessons/load-mnist-dataset-from-pytorch-torchvision
# [2] https://nextjournal.com/gkoehler/pytorch-mnist
# [3] https://www.codementor.io/@dejanbatanjac/pytorch-the-missing-manual-on-loading-mnist-dataset-wjeh5top7
#     -> here instructions about cuda (not implemented in this project yet)

import argparse
from matplotlib import rcParams
import matplotlib.pyplot as plt
from numpy import column_stack, loadtxt, savetxt
from os.path import exists
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
n_epochs = 2 # tmp, change to 10

t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0), std=(1))])

if __name__ == '__main__':

  rcParams['font.size'] = 24
  rcParams['legend.fontsize'] = 16

  parser = argparse.ArgumentParser(description='Analyze coincidences file and plot results of the analysis',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-m', '--mode',
                      dest='mode',
                      type=str,
                      default='calculate',
                      help='mode of the script: calculate, plot')

  args = parser.parse_args()

  if args.mode == "calculate":

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
      losses, counters = train(network, epoch, train_loader, optimizer)
      train_losses += losses
      train_counter += counters
      loss = test(network, test_loader)
      test_losses.append(loss)

    # Saving losses data to files:
    savetxt(results_subdirectory + 'train_losses.txt', column_stack((train_counter, train_losses)))
    savetxt(results_subdirectory + 'test_losses.txt', column_stack((test_counter, test_losses)))

  elif args.mode == "plot":

    train_results_path = results_subdirectory + 'train_losses.txt'
    test_results_path = results_subdirectory + 'test_losses.txt'

    if exists(train_results_path) and exists(test_results_path):

      train_results = loadtxt(train_results_path)
      test_results = loadtxt(test_results_path)

      fig = plt.figure(figsize=(8, 6))
      ax = fig.add_subplot(111)
      plt.subplots_adjust(left=0.16, right=0.96, top=0.91, bottom=0.17)
      plt.plot(train_results[:,0], train_results[:,1], label="train loss")
      plt.plot(test_results[:,0], test_results[:,1], 'o', label="test loss")
      plt.title("Losses")
      plt.xlabel("Counter")
      plt.ylabel("Loss")
      plt.legend(loc=1)
      #plt.ylim(ymin=0)
      plt.savefig(results_subdirectory + "losses.png")
      plt.clf()

    else:

      print("Firstly use the calculate mode: ./mnist.py --m calculate")

  else:

    print("To check available modes run command: ./mnist.py -h")

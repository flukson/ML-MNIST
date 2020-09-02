#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torchvision.datasets as datasets

data_subdirectory = "./data/"

if __name__ == '__main__':

  # https://www.aiworkbox.com/lessons/load-mnist-dataset-from-pytorch-torchvision
  print("1. Loading MNIST dataset")
  mnist_trainset = datasets.MNIST(root=data_subdirectory, train=True, download=True, transform=None)
  mnist_testset = datasets.MNIST(root=data_subdirectory, train=False, download=True, transform=None)
  print()


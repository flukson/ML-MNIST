#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

data_subdirectory = "./data/"

bs = 512

t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0), std=(1))])

if __name__ == '__main__':

  # https://www.aiworkbox.com/lessons/load-mnist-dataset-from-pytorch-torchvision
  mnist_trainset = datasets.MNIST(root=data_subdirectory, train=True, download=True, transform=t)
  mnist_testset = datasets.MNIST(root=data_subdirectory, train=False, download=True, transform=t)

  # https://www.codementor.io/@dejanbatanjac/pytorch-the-missing-manual-on-loading-mnist-dataset-wjeh5top7
  dl_train = DataLoader(mnist_trainset, batch_size=bs, drop_last=True, shuffle=True)
  dl_train = DataLoader(mnist_testset, batch_size=bs, drop_last=True, shuffle=True)

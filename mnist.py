#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from modules import prepare_data

data_subdirectory = "./data/"

if __name__ == '__main__':

  print("1. Downloading data:")
  prepare_data.downloadData(data_subdirectory)
  print()

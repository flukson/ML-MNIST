import gzip
from os.path import exists
import requests
import shutil
import sys

from . common import *

def downloadData(data_subdirectory, force=False):

  for data_file in data_files:

    data_file_path = data_subdirectory + data_file

    if not exists(data_file_path) or force:
      print("Downloading " + data_file + "...")
      r = requests.get(website_url + data_file, allow_redirects=True)
      open(data_file_path, 'wb').write(r.content)
    else:
      print("File " + data_file + " already downloaded.")

def unpackData(data_subdirectory, force=False):

  for data_file in data_files:

    data_file_path = data_subdirectory + data_file

    if not exists(data_file_path[0:-3]) or force:
      print("Unpacking " + data_file + "...")
      with gzip.open(data_file_path, 'rb') as f_in:
        with open(data_file_path[0:-3], 'wb') as f_out:
          shutil.copyfileobj(f_in, f_out)
    else:
      print("File " + data_file + " already unpacked.")

from os.path import exists
import requests
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

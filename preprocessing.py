import os
import wget
import zipfile

import utils

def download_data(params):
  
  url_tar_file = params['url_dataset']

  if not os.path.exists(params['data_dir']):
    os.makedirs(params['data_dir'])

  wget.download(url_tar_file, params['data_dir'])

def extract_data(params):
    
  tar_file = os.path.join(params['data_dir'],params['compressed_data_name'])
  
  zip_ref = zipfile.ZipFile(tar_file, 'r')
  zip_ref.extractall(params['data_dir'])
  zip_ref.close()


if __name__ == '__main__':

  params = utils.yaml_to_dict('config.yml')
  download_data(params)
  extract_data(params)


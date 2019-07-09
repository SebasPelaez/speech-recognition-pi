import os
import wget
import zipfile

from srpi.utils import folder
def download_data(params):
  
  url_tar_file = params['url_dataset']
  data_dir = params['data_dir']
  data_dir_images = os.path.join(data_dir,params['data_dir_images'])
  data_dir_fragments = os.path.join(data_dir, params['data_dir_fragments'])

  folder.create_multiple([data_dir, data_dir_fragments, data_dir_images])
  wget.download(url_tar_file, params['data_dir'])

def extract_data(params):
    data_dir =  params['data_dir']
    compressed_data_name = params['compressed_data_name']

    tar_file = os.path.join(data_dir,  compressed_data_name)
  
    zip_ref = zipfile.ZipFile(tar_file, 'r')
    zip_ref.extractall(data_dir)
    zip_ref.close()
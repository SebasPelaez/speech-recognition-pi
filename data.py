import os

import pandas as pd

from tensorflow import keras

def make_datagenerator(params, mode):
    
  file_path = mode + '.txt'
  data_path = os.path.join(params['data_dir'],file_path)
    
  df = pd.read_csv(data_path, sep="\t", header=0)
    
  datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
  generator = datagen.flow_from_dataframe(
    dataframe = df,
    x_col = 'images',
    y_col = 'labels',
    target_size = params['image_shape'],
    batch_size = params['batch_size'],
    class_mode = params['class_mode'],
    shuffle=params['shuffle']
  )
   
  return generator
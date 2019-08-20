import tensorflow as tf

import argparse
import json
import keras
import os

import data
import model
import preprocessing
import utils

def train_model(params):

  print('Data in train')
  train_generator = data.make_datagenerator(params, mode='training')
  print('Data in validation')
  val_generator = data.make_datagenerator(params,mode='validation')

  width, height = params["image_shape"]

  inputs = tf.keras.layers.Input(shape=(width, height, 3))
  net = model.ModelArchitecture(num_classes=params['num_classes'])
  net(inputs, training=False)

  cp_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(params['model_dir'], 'tf_ckpt'), 
    save_weights_only=True, 
    verbose=1,
    period=5)

  tb_callback = tf.keras.callbacks.TensorBoard(
    os.path.join(params['model_dir'], 'logs'))

  optimizer = tf.keras.optimizers.Adam(lr=params['learning_rate'])

  steps_per_epoch = train_generator.n // params['batch_size']
  validation_steps = val_generator.n // params['batch_size']

  net.compile(optimizer=optimizer, loss=params['loss'], metrics=['sparse_categorical_accuracy'])
  net.fit_generator(
    train_generator, 
    steps_per_epoch=steps_per_epoch, 
    epochs=params['num_epochs'],
    workers=4,
    validation_data=val_generator, 
    validation_steps=validation_steps,
    callbacks=[cp_callback,tb_callback])

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')
  parser.add_argument('-v', '--verbosity', default='INFO',
    choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARM'],
  )
  
  args = parser.parse_args()
  tf.logging.set_verbosity(args.verbosity)

  params = utils.yaml_to_dict(args.config)
  tf.logging.info("Using parameters: {}".format(params))
  train_model(params)
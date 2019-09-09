from datetime import datetime, timedelta

import argparse
import os
import speech_recognition as sr

import preprocessing
import utils

import model
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img



TARGET_NAMES = ["acelera", "arranca", "avanza", "derecha",
                "pita", "izquierda", "frena", "detente", "retrocede", "gira"]

parser = argparse.ArgumentParser(description="Pipeline for keywords predicts")
parser.add_argument("--wait", type=int, default="", help="Waiting time")
args = parser.parse_args()
wait = args.wait


recognizer = sr.Recognizer()
recognizer.energy_threshold = 7000
recognizer.non_speaking_duration = 0.3
recognizer.pause_threshold = 0.3
recognizer.dynamic_energy_adjustment_ratio = 2
microphone = sr.Microphone()

next_time = datetime.now() - timedelta(seconds=1)
current_time = datetime.now().time()

params = utils.yaml_to_dict('config.yml')
params['data_dir'] = './microphone-dataset'
params['save_audio_fragments'] = False
params['model_dir'] = params['model_dir']



def load_model(params):
    tf.keras.backend.clear_session()
    width, height = params['image_shape']
    inputs = tf.keras.layers.Input(shape=(width, height, 3))
    net = model.ModelArchitecture(num_classes=params['num_classes'])
    x = net(inputs, training=False)
    return net

try:
  NET = load_model(params) 
  optimizer = tf.keras.optimizers.Adam(lr=params['learning_rate'])
  NET.compile(optimizer=optimizer, loss=params['loss'], metrics=[
              'sparse_categorical_accuracy'])
  NET.load_weights(os.path.join(params['model_dir'], 'tf_ckpt'))
  print("[INFO] Model loaded")
except :
  print("[ERROR] Some error while try to load model")

def make_predictions(image):
    predictions = NET.predict(image, batch_size=1)
    print("Prediction", predictions)
    return np.argmax(predictions, axis=1)


def listen():
  """"""
  global i
  with microphone as source:
    audio = recognizer.listen(source)

  # write audio to a WAV file
  filename = './microphone-dataset/audios/microphone-result.wav'
  specgram_image = "./microphone-dataset/images/audios/specgram_matrix_microphone-result_segment0.png"
  
  with open(filename, "wb") as f:
    try:
      f.write(audio.get_wav_data())
      print('[INFO] Saving ', audio)
    except:
      print("[ERROR] Some error in saved-audio script")

  try:
    print("[INFO] Preprocesing audio script")
    preprocessing.generate_spectogram_images(params)
  except :
    print("[ERROR]: Some error in preprocessing-audio script")
  

  img = load_img(specgram_image)
  img = img.resize((110, 480))
  img = np.array(img)
  img = img[np.newaxis, ...]
  print("[INFO] Size of matrix :{} ".format(img.shape))

  try:
    prediction = make_predictions(img)
    print("[INFO] Predictions {}".format(prediction))
    print("[INFO] Talk!")
  except :
    print("[ERROR] We can`t predict ")
print("[INFO] Talk!")

while True:
  current_time = datetime.now()
  if(current_time >= next_time):
    listen()
    next_time = datetime.now() + timedelta(seconds=wait)

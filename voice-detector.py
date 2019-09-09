from datetime import datetime, timedelta

import argparse
import os
import speech_recognition as sr

import preprocessing
import utils

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

def listen():
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
  


while True:
  current_time = datetime.now()
  if(current_time >= next_time):
    listen()
    next_time = datetime.now() + timedelta(seconds=wait)

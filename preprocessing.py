import cv2
import json
import os
import scipy.io.wavfile
import wget
import zipfile

import utils

import numpy as np
import pandas as pd

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioSegmentation

from sklearn.preprocessing import MinMaxScaler

def download_data(params):

  url_tar_file = params['url_dataset']

  if not os.path.exists(params['data_dir']):
    os.makedirs(params['data_dir'])
    os.makedirs(os.path.join(params['data_dir'],params['data_dir_fragments']))
    os.makedirs(os.path.join(params['data_dir'],params['data_dir_images']))

  wget.download(url_tar_file, params['data_dir'])

def extract_data(params):

  tar_file = os.path.join(params['data_dir'],params['compressed_data_name'])

  zip_ref = zipfile.ZipFile(tar_file, 'r')
  zip_ref.extractall(params['data_dir'])
  zip_ref.close()

def generate_spectogram_images(params):

  audios_path = os.path.join(params['data_dir'],params['data_dir_audios'])
  for root, dirs, files in os.walk(audios_path, topdown=False):

    for name in files:

      fragment_folder = os.path.join(params['data_dir'],params['data_dir_fragments'],root.split(os.path.sep)[-1])
      specgram_folder = os.path.join(params['data_dir'],params['data_dir_images'],root.split(os.path.sep)[-1])

      [fs, x] = audioBasicIO.readAudioFile(os.path.join(root, name))

      x = audioBasicIO.stereo2mono(x)
      x = _rescaled_signal(x)

      segments = _find_segments_from_audio(x=x, fs=fs)

      for i,segment in enumerate(segments):

        imname = 'specgram_matrix_' + os.path.splitext(name)[0] + '_segment' + str(i) + '.png'
        audio_fragment = _extract_audio_fragments(x=x, fs=fs, segment=segment)

        if params['save_audio_fragments']:
          _save_audio_fragments(fragment_folder, audio_fragment, fs, name, i)

        trainable_frames = _extract_trainable_frames(audio_fragment, imname, params)

        if trainable_frames is not None:

          specgram_list = list()
          for frame in trainable_frames:

            specgram, TimeAxis, FreqAxis = audioFeatureExtraction.stSpectogram(
                frame,
                fs,
                round(fs * 0.02),
                round(fs * 0.01),
                False
            )

            specgram = np.expand_dims(specgram,axis=2)
            specgram_list.append(specgram)

          specgram_matrix = np.concatenate((specgram_list),axis=2)
          specgram_matrix = specgram_matrix * 255
          _save_specgram_as_image(specgram_folder, specgram_matrix, imname)

def _enframe(x, winlen, hoplen, frames, imname):
  '''
  receives a 1D numpy array and divides it into frames.
  outputs a numpy matrix with the frames on the rows.
  '''
  x = np.squeeze(x)
  if len(x) <= winlen:
    print('Window size is bigger than record segment: {}'.format(imname))
    return None
  if x.ndim != 1: 
    print('Enframe input segments: {} must be a 1-dimensional array'.format(imname))
    return None

  n_frames = 1 + np.int(np.floor((len(x) - winlen) / float(hoplen)))

  while n_frames < frames:
    hoplen = int(hoplen * 0.8)
    n_frames = 1 + np.int(np.floor((len(x) - winlen) / float(hoplen)))


  xf = np.zeros((n_frames, winlen))
  for ii in range(n_frames):
    xf[ii] = x[ii * hoplen : ii * hoplen + winlen]

  return xf

def _extract_trainable_frames(audio_fragment, imname, params):

  frames = params['frames']
  windows_matrix = _enframe(audio_fragment,params['window_length'],params['shift_frames'],frames,imname)

  if windows_matrix is not None:

    middle_frames = frames // 2

    windows_matrix_len = len(windows_matrix)

    if frames > windows_matrix_len:
      raise TypeError("There are not enough frames")

    if frames < windows_matrix_len:

      if (frames%2) == 0:
        lower_lim = (windows_matrix_len//2) - (middle_frames - 1)
      else:
        lower_lim = (windows_matrix_len//2) - middle_frames

      upper_lim = (windows_matrix_len//2) + middle_frames
      trainable_frames = windows_matrix[lower_lim-1:upper_lim,:]

    else:
      trainable_frames = windows_matrix

    return trainable_frames

  return None

def _save_audio_fragments(fragment_folder, audio_fragment, fs, name, index_name):

  if not os.path.exists(fragment_folder):
    os.makedirs(fragment_folder)

  name_wav = 'audio_fragment_' + os.path.splitext(name)[0] + '_segment' + str(index_name) + '_.wav'

  scipy.io.wavfile.write(
    os.path.join(fragment_folder,name_wav),
    fs,
    audio_fragment)

def _save_specgram_as_image(specgram_folder, specgram_matrix, imname):

  if not os.path.exists(specgram_folder):
    os.makedirs(specgram_folder)

  fpath = os.path.join(specgram_folder, imname)
  cv2.imwrite(fpath, specgram_matrix)

def _rescaled_signal(x):

  scaler = MinMaxScaler()

  reshaped_signal = np.reshape(x,(-1,1)).astype(float)
  rescaled_signal = scaler.fit_transform(reshaped_signal)

  original_shape = np.reshape(rescaled_signal,(-1,))

  return original_shape

def _find_segments_from_audio(x, fs):

  segments = audioSegmentation.silenceRemoval(
    x,
    fs,
    0.020,
    0.020,
    smoothWindow = 1.0,
    weight = 0.3,
    plot = False)

  return segments

def _extract_audio_fragments(x, fs, segment):

  lower_lim, upper_lim = segment

  lower_lim = int(lower_lim*fs)
  upper_lim = int(upper_lim*fs)

  audio_segment = x[lower_lim:upper_lim]

  return audio_segment

def make_id_label_map(params):

  images_path = os.path.join(params['data_dir'], params['data_dir_images'])
  labels_json_path = os.path.join(params['data_dir'],params['label_id_json'])
  id_label_map = dict()

  for i,value in enumerate(os.listdir(images_path)):
    id_label_map[value] = i

  with open(labels_json_path,'w') as file:
    json.dump(id_label_map,file)

def _load_label_id_map(params):

  label_json_path = os.path.join(params['data_dir'], params['label_id_json'])
  with open(label_json_path, 'r') as file:
    label_dict = json.load(file)

  labels_json_dict = {v:k for k,v in enumerate(label_dict)}

  return labels_json_dict

def split_data(params):

  images_path = os.path.join(params['data_dir'], params['data_dir_images'])

  label_id_map = _load_label_id_map(params)

  images_list = list()
  label_list = list()

  for root, dirs, files in os.walk(images_path, topdown=False):
    for name in files:
      image_name = os.path.join(root, name)
      label_class = image_name.split(os.sep)[-2]

      images_list.append(image_name)
      label_list.append(label_id_map[label_class])

  all_data = pd.DataFrame.from_dict({'images':images_list,'labels':label_list})

  training_df = pd.DataFrame()
  validation_df = pd.DataFrame()
  test_df = pd.DataFrame()

  for label in np.arange(params['num_classes']):

    df_group = all_data[all_data['labels'] == label]

    training_sample = df_group.sample(frac=0.98)
    validation_sample = df_group.drop(training_sample.index).sample(frac = 0.5)
    test_sample = df_group.drop(training_sample.index).drop(validation_sample.index)

    training_df = training_df.append(training_sample)
    validation_df = validation_df.append(validation_sample)
    test_df = test_df.append(test_sample)

  training_df.to_csv(os.path.join(params['data_dir'],params['training_data']), header=True, index=None, sep='\t')
  validation_df.to_csv(os.path.join(params['data_dir'],params['validation_data']), header=True, index=None, sep='\t')

  test_df.to_csv(os.path.join(params['data_dir'],params['test_data']), header=True, index=None, sep='\t')

if __name__ == '__main__':

  params = utils.yaml_to_dict('config.yml')
  download_data(params)
  extract_data(params)
  generate_spectogram_images(params)
  make_id_label_map(params)
  split_data(params)
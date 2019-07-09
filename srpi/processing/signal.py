import numpy as np
from matplotlib import pyplot as plt
from skimage import util
from sklearn.preprocessing import MinMaxScaler

def slice_signal(audio, win_shape, step=100):
    win_shape = (win_shape, )
    slices = util.view_as_windows(audio, win_shape, step)
    print(f'Audio shape: {audio.shape}, Sliced audio shape: {slices.shape}')
    return slices

def windowing_signal(M, slices):
    win = np.hanning(M + 1)[:-1]
    slices = slices * win
    slices = slices.T
    print('Shape of `slices` :', slices.shape)
    return slices

def  get_spectrum(slices, M):
    spectrum = np.fft.fft(slices, axis=0 )[:M//2 + 1 : -1]
    return np.abs(spectrum)

def rescaled_signal(signal):
    
  scaler = MinMaxScaler()
  
  reshaped_signal = np.reshape(signal,(-1,1)).astype(float)
  rescaled_signal = scaler.fit_transform(reshaped_signal)

  original_shape = np.reshape(rescaled_signal,(-1,))
  
  return original_shape

def plot_spectrum( spectrum, L , rate):
    f, ax = plt.subplots(figsize=(4.2 * 2, 2.4 *2))
    S = np.abs(spectrum)
    ax.imshow(S, origin= 'lower', cmap='viridis', extent=(0,L, 0 , rate / 2 / 1000))
    ax.axis('tight')
    ax.set_ylabel('Frecuency [kHz]')
    ax.set_xlabel('Time [s]')

    
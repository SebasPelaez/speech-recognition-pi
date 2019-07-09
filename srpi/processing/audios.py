import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile

def load_audio(file_url):
    rate, audio = wavfile.read(file_url)
    return rate, audio

def stereo2mono(audio):
    return np.mean(audio, axis=1)

def plot_audio(rate, audio):
    N = audio.shape[0]
    L = N / rate

    seconds = np.arange(N) / rate
    
    print(f'Audio Length: {L:.2f} seconds')
    f, ax = plt.subplots()
    ax.plot(seconds, audio)



    
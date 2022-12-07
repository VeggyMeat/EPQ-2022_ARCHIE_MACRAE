import os
# disables tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
import tensorflow_io as tfio
import numpy as np
import soundfile as sf

def to_spectogram(data_file_loc):
    # reads the data out 
    data = tf.io.read_file(data_file_loc)

    # decodes the audio out of flac format
    audio = tfio.audio.decode_flac(data, dtype=tf.int16)

    # deletes redundant data in the encoding
    audio = tf.squeeze(audio, axis=-1)

    # changes the data type to a float (able to handle non integers)
    audio = tf.cast(audio, tf.float32)

    # actually creates the spectrogram
    spectrogram = tf.signal.stft(audio, frame_length=256, frame_step=160, fft_length=384)

    # returns only the magnitude and gets rid of redundant scaling
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)

    # normalises the data
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    return spectrogram

print(to_spectogram("/home/amri123/Desktop/Training Data/Data/0000000.flac"))
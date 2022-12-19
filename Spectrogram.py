import os
# disables tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
import tensorflow_io as tfio
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from Constants import *
import time

def to_spectrogram(data_file_loc):
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


def display_spectrogram(spectrogram):
    fig = plt.figure(figsize=(8,5))
    ax = plt.subplot(2, 1, 2)
    spectrogram = np.array([np.trim_zeros(x) for x in np.transpose(spectrogram)])
    ax.imshow(spectrogram, vmax=1)
    ax.axis("off")
    plt.show()


def file_to_spectrogram(file_loc_in, file_loc_out):
    # gets the spectrograms
    spectrogram = to_spectrogram(file_loc_in)

    # converts it to a numpy file
    numpy_spect = spectrogram.numpy()

    # saves it to the output file
    np.savez_compressed(file_loc_out, numpy_spect)


def read_spectrogram_file(file_loc):
    # reads it in as a numpy file
    numpy_spect = np.load(file_loc)['arr_0']

    # converts it to a tensorflow tensor
    return tf.convert_to_tensor(numpy_spect, dtype=tf.float32)


def time_test_spectrogram():
    start_time = time.time()

    a = [to_spectrogram("/home/amri123/Desktop/Training Data/Data/000000" + str(x) + ".flac") for x in range(10)]

    print(time.time() - start_time)

    print(a)


def spectrogram_files_conversion(dir_in, dir_out, n):
    start_time = time.time()
    # repeats n times
    one_percent = n // 100
    for counter in range(n):
        # uses the file to spectrogram with the directories with laeding 0 files
        file_to_spectrogram(os.path.join(dir_in, str(counter).zfill(7) + ".flac"), os.path.join(dir_out, str(counter).zfill(7) + ".npz"))

        if counter % one_percent == 0:
                # prints an estimated time and percentage done message
                if counter != 0:
                    percent_done = counter // one_percent
                    print(str(percent_done) + " percent done")
                    time_elapsed = time.time() - start_time
                    print("estimated " + str((time_elapsed / percent_done) * (100 - percent_done) / 60)[:7] + " minutes left", end='\n\n')

    print(str(time.time() - start_time) + " seconds taken")


# spectrogram_files_conversion("/home/amri123/Desktop/Training Data/Data", "/media/amri123/External SSD/Spectrograms", 252702)

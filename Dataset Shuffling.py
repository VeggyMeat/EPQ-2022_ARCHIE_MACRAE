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
import random


def to_npz(dir, n):
    return os.path.join(dir, str(n).zfill(7) + '.npz')


def to_txt(dir, n):
    return os.path.join(dir, str(n).zfill(7) + '.txt')


def data_shuffle(label_dir, spectrogram_dir, n):
    prev_time = time.time()

    made = [x for x in range(n)]
    for x in range(n):
        # gets a random index from made, and removes it from made
        item = made.pop(random.randint(0, len(made) - 1))

        # renames .txt and .flac file to temporary files
        os.rename(to_txt(label_dir, item), to_txt(label_dir, x + n))
        os.rename(to_npz(spectrogram_dir, item), to_npz(spectrogram_dir, x + n))

        # renames old temp variables to new file names
        if x != 0:
            os.rename(to_txt(label_dir, x + n - 1), to_txt(label_dir, item))
            os.rename(to_npz(spectrogram_dir, x + n - 1), to_npz(spectrogram_dir, item))
        else:
            first = item

    # does the final conversion of the last temporary file
    os.rename(to_txt(label_dir, x + n), to_txt(label_dir, first))
    os.rename(to_npz(spectrogram_dir, x + n), to_npz(spectrogram_dir, first))

    print(time.time() - prev_time)

data_shuffle('/media/amri123/External SSD/Labels', '/media/amri123/External SSD/Spectrograms', 252702)

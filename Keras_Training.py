import Label_Handling
import Spectrogram
import tensorflow as tf
from CNN_RNN_Model import model_creation, CTCLoss
from tensorflow import keras
import os
from jiwer import wer
import time
import numpy as np

model = model_creation(193, len(Label_Handling.chars))

opt = keras.optimizers.Adam(learning_rate=1e-4)

model.compile(optimizer=opt, loss=CTCLoss)

batch_size = 12
total_files = 252702
epochs = 50
spectrogram_dir = "/media/amri123/External SSD/Data"
labels_dir = "/media/amri123/External SSD/Labels"

files_train = 120
files_validate = 120

train_spectrograms = [os.path.join(spectrogram_dir, str(x).zfill(7) + '.flac') for x in range(files_train)]
validate_spectrograms = [os.path.join(spectrogram_dir, str(x).zfill(7) + '.flac') for x in range(files_train, files_train + files_validate)]

train_labels = [os.path.join(labels_dir, str(x).zfill(7) + '.txt') for x in range(files_train)]
validate_labels = [os.path.join(labels_dir, str(x).zfill(7) + '.txt') for x in range(files_train, files_train + files_validate)]

def get_data(file_loc, label_loc):
    spectrogram = Spectrogram.to_spectrogram(file_loc)
    label = Label_Handling.read_num_file(label_loc)

    return spectrogram, label

train_dataset = tf.data.Dataset.from_tensor_slices((train_spectrograms, train_labels))
train_dataset = (train_dataset.map(get_data, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))

validation_dataset = tf.data.Dataset.from_tensor_slices((validate_spectrograms, validate_labels))
validation_dataset = (validation_dataset.map(get_data, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))

start_time = time.time()

history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)

print(time.time() - start_time)

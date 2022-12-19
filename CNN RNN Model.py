import os
# disables tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
import tensorflow_io as tfio
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from jiwer import wer
from keras import layers
import Label_Handling
import Spectrogram
import Label_Handling


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


def model_creation(input_values, ouptput_values):
    # the input layer
    spectrogram_in = layers.Input((None, input_values), name="input_layer")
    
    # layer which changes the input layer to a 2D array
    model = layers.Reshape((-1, input_values, 1), name="2D_expand_reshape")(spectrogram_in)

    # first convolutional layer
    model = layers.Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False, name="CNN_1")(model)
    model = layers.BatchNormalization(name="CNN_1_bn")(model)
    model = layers.ReLU(name = "CNN_1_relu")(model)

    # second convolutional layer
    model = layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False, name="CNN_2")(model)
    model = layers.BatchNormalization(name="CNN_2_bn")(model)
    model = layers.ReLU(name = "CNN_2_relu")(model)

    # third convolutional layer
    model = layers.Conv2D(filters=64, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False, name="CNN_3")(model)
    model = layers.BatchNormalization(name="CNN_3_bn")(model)
    model = layers.ReLU(name = "CNN_3_relu")(model)

    # reshapes the volume to go into the RNN layers
    model = layers.Reshape((-1, model.shape[-2] * model.shape[-1]))(model)

    # creates the reccurent layers
    for x in range(6):
        reccurent = layers.GRU(units=800, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, reset_after=True, name="GRU_" + str(x + 1))

        model = layers.Bidirectional(reccurent, merge_mode="concat", name="bidirectional_" + str(x + 1))(model)

        # adds a dropout layer for better training
        model = layers.Dropout(rate=0.5)(model)

    # adds two fully connected dense layers
    model = layers.Dense(units=1600, name="dense_1")(model)
    model = layers.ReLU(name="dense_1_relu")(model)
    model = layers.Dense(units=1600, name="dense_2")(model)
    model = layers.ReLU(name="dense_2_relu")(model)

    # output layer
    model = layers.Dense(units=ouptput_values + 1, activation="softmax")(model)

    # classifying the final model
    model = tf.keras.Model(spectrogram_in, model, name="DeepSpeech_2")

    return model


model = model_creation(193, len(Label_Handling.chars))

optimiser = tf.keras.optimizers.Adam()

batch_size = 12
total_files = 252702
epochs = 1
spectrogram_dir = "/media/amri123/External SSD/Spectrograms"
labels_dir = "/media/amri123/External SSD/Labels"

files_train = 12000
files_validate = 120

train_spectrograms = [os.path.join(spectrogram_dir, str(x).zfill(7) + '.npz') for x in range(files_train)]
validate_spectrograms = [os.path.join(spectrogram_dir, str(x).zfill(7) + 'npz') for x in range(files_train, files_train + files_validate)]

train_labels = [os.path.join(labels_dir, str(x).zfill(7) + '.txt') for x in range(files_train)]
validate_labels = [os.path.join(labels_dir, str(x).zfill(7) + 'txt') for x in range(files_train, files_train + files_validate)]

# model.summary()


num = 0

print("huh")

for epoch in range(epochs):
    epoch_loss_average = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for batch in range(files_train // batch_size):
        print("ok")
        labels = []
        spectrograms = []
        for file in range(12):
            labels.append(Label_Handling.read_num_file(train_labels[num]))
            spectrograms.append(Spectrogram.read_spectrogram_file(train_spectrograms[num]))
        
        labels = tf.convert_to_tensor(labels, dtype=tf.int64)
        spectrograms = tf.convert_to_tensor(spectrograms, dtype=tf.float32)

        predicted = model(spectrograms, training=True)

        loss = CTCLoss(labels, predicted)
        with tf.GradientTape() as tape:
            gradients = tape.gradient(loss, model.trainable_variables)
        
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))

        epoch_loss_average.update_state(loss)
        epoch_accuracy.update_state(labels, predicted)

        num += 1

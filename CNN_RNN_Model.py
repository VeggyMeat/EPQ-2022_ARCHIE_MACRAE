import os
# disables tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
import tensorflow_io as tfio
import tensorflow_datasets as tfds
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from keras import layers
import Label_Handling


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


def model_creation(input_values, ouptput_values, dropout_freq=0.5):
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
    # model = layers.Conv2D(filters=64, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False, name="CNN_3")(model)
    # model = layers.BatchNormalization(name="CNN_3_bn")(model)
    # model = layers.ReLU(name = "CNN_3_relu")(model)

    # reshapes the volume to go into the RNN layers
    model = layers.Reshape((-1, model.shape[-2] * model.shape[-1]))(model)

    # creates the reccurent layers
    for x in range(5):
        reccurent = layers.GRU(units=512, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, reset_after=True, name="GRU_" + str(x + 1))

        model = layers.Bidirectional(reccurent, merge_mode="concat", name="bidirectional_" + str(x + 1))(model)

        # adds a dropout layer for better training
        if x != 4:
            model = layers.Dropout(rate=dropout_freq)(model)

    # adds two fully connected dense layers
    model = layers.Dense(units=1600, name="dense_1")(model)
    model = layers.ReLU(name="dense_1_relu")(model)
    model = layers.Dropout(rate=dropout_freq)(model)
    # model = layers.Dense(units=1024, name="dense_2")(model)
    # model = layers.ReLU(name="dense_2_relu")(model)
    # model = layers.Dropout(rate=dropout_freq)(model)

    # output layer
    output = layers.Dense(units=ouptput_values + 1, activation="softmax")(model)

    # classifying the final model
    Model = tf.keras.Model(spectrogram_in, output, name="DeepSpeech_2")

    return Model

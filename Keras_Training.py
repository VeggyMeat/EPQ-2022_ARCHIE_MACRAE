import Label_Handling
import Spectrogram
import tensorflow as tf
from CNN_RNN_Model import model_creation, CTCLoss
from tensorflow import keras
import os
from jiwer import wer
import time
import numpy as np

model = model_creation(193, len(Label_Handling.chars) + 1)

# change to  stochastic gradient descent

opt = keras.optimizers.Adam(learning_rate=1e-4)

model.compile(optimizer=opt, loss=CTCLoss)

batch_size = 12
total_files = 252702
epochs = 50
spectrogram_dir = "/media/amri123/External SSD/Data"
labels_dir = "/media/amri123/External SSD/Labels"

debug_dir = "/media/amri123/External SSD/Debug"

files_train = 12000
files_validate = 12

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


def decode_CTC(nums):
    out = []
    prev = nums[0]
    blank = len(Label_Handling.chars) + 1
    for num in nums:
        if num != prev:
            if prev != blank:
                out.append(prev)
        prev = num

    if prev != blank:
        out.append(prev)
    return out


# https://stackoverflow.com/questions/48118111/get-loss-values-for-each-training-instance-keras


class SentenceInfo(keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            x, y = batch
            batch_predictions = model.predict(x)
            for item in batch_predictions:
                nums = []
                for ls in item:
                    nums.append(tf.math.argmax(ls).numpy())
                results = decode_CTC(nums)
                chars = ""
                for num in results:
                    chars += Label_Handling.inv_map[num]
                predictions.append(chars)
            
            for item in y:
                sentence = ""
                for char in item:
                    sentence += Label_Handling.inv_map[char.numpy()]
                targets.append(sentence)
        
        error = wer(targets, predictions) 
        file = open(os.path.join(debug_dir, str(epoch).zfill(3) + ".txt"), 'w')
        file.write("WER: " + str(error) + '\n')
        print(error)
        for i in range(len(targets)):
            print('\n', predictions[i])
            print(targets[i], '\n')
            file.write("\ntarget: " + targets[i] + '\n')
            file.write("predicted: " + predictions[i] + '\n\n')
        file.close()


history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=[SentenceInfo(validation_dataset)])

print(time.time() - start_time)

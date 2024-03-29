import Label_Handling
import Spectrogram
import tensorflow as tf
from CNN_RNN_Model import model_creation, CTCLoss
from tensorflow import keras
import os
from jiwer import wer
import time
import numpy as np
import Dataset_Shuffling

start_time = time.time()

# shuffles the dataset files
# Dataset_Shuffling.data_shuffle_flac('/media/amri123/External SSD/Labels', '/media/amri123/External SSD/Data', 252702)
Dataset_Shuffling.data_shuffle_flac('/media/amri123/External SSD/Labels2', '/media/amri123/External SSD/Data2', 104014)

# creates the initial files for storing the losses the WER

validation_loss_file = "/media/amri123/External SSD/validation_loss.txt"
loss_file = "/media/amri123/External SSD/loss.txt"
WER_file = "/media/amri123/External SSD/WER.txt"
time_file = "/media/amri123/External SSD/time.txt"

# creates the files to store the data and clears them

file = open(validation_loss_file, 'w')
file.close()
file = open(loss_file, 'w')
file.close()
file = open(WER_file, 'w')
file.close()

model = model_creation(193, len(Label_Handling.chars) + 1, dropout_freq=0.5)

model.summary()

# adam optimiser
# opt = keras.optimizers.Adam(learning_rate=1e-4)

opt = keras.optimizers.Adamax(learning_rate=1e-4)

model.compile(optimizer=opt, loss=CTCLoss)

batch_size = 10
# total_files = 252702
total_files = 104014
epochs = 50

# defines directories for the output files

spectrogram_dir = "/media/amri123/External SSD/Data2"
labels_dir = "/media/amri123/External SSD/Labels2"
debug_dir = "/media/amri123/External SSD/Debug"

# creates the debug directory if it does not already exist

if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)

# defines the number of files to train on

files_train = 12000
files_validate = 600

# sets up the file names for all the spectrograms and labels

train_spectrograms = [os.path.join(spectrogram_dir, str(x).zfill(7) + '.flac') for x in range(files_train)]
validate_spectrograms = [os.path.join(spectrogram_dir, str(x).zfill(7) + '.flac') for x in range(files_train, files_train + files_validate)]

train_labels = [os.path.join(labels_dir, str(x).zfill(7) + '.txt') for x in range(files_train)]
validate_labels = [os.path.join(labels_dir, str(x).zfill(7) + '.txt') for x in range(files_train, files_train + files_validate)]

# creates a function to take in a spectrogram and label file, and output their data in the right format

def get_data(file_loc, label_loc):
    spectrogram = Spectrogram.to_spectrogram(file_loc)
    label = Label_Handling.read_num_file(label_loc)

    return spectrogram, label

# creates the tensorflow dataset

train_dataset = tf.data.Dataset.from_tensor_slices((train_spectrograms, train_labels))
train_dataset = (train_dataset.map(get_data, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))

validation_dataset = tf.data.Dataset.from_tensor_slices((validate_spectrograms, validate_labels))
validation_dataset = (validation_dataset.map(get_data, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))

# decodes the CTC to the characters

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

# the callback class that generates examples of the network's response

class SentenceInfo(keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        # decodes sample data from validation set and the model's prediction
        predictions = []
        targets = []
        losses = []
        for batch in self.dataset:
            x, y = batch
            batch_predictions = model.predict(x)

            losses.append(sum(CTCLoss(y, batch_predictions).numpy()) / batch_size)

            # goes through each prediction in the batch and decodes it into the characters out
            for item in batch_predictions:
                nums = []
                for ls in item:
                    nums.append(tf.math.argmax(ls).numpy())
                results = decode_CTC(nums)
                chars = ""
                for num in results:
                    chars += Label_Handling.inv_map[num]
                predictions.append(chars)
            
            # translates the labels back into the sentances
            for item in y:
                sentence = ""
                for char in item:
                    sentence += Label_Handling.inv_map[char.numpy()]
                targets.append(sentence)

        # calculates WER, prints and logs the outputs of the information out
        
        error = wer(targets, predictions)
        file = open(os.path.join(debug_dir, str(epoch).zfill(3) + ".txt"), 'w')
        file.write("WER: " + str(error) + '\n')

        print("WER: " + str(error))

        for i in range(len(targets)):
            print('\n target:', targets[i])
            print('predictions: ' + predictions[i], '\n')
            file.write("\ntarget: " + targets[i] + '\n')
            file.write("predicted: " + predictions[i] + '\n\n')

        file.close()

        # write loss and WER out

        file = open(loss_file, 'a')
        file.write(str(logs['loss']) + '\n')
        file.close()

        file = open(WER_file, 'a')
        file.write(str(error) + '\n')
        file.close()

        # writes validation loss

        validation_loss = (sum(losses) / len(losses))[0]

        file = open(validation_loss_file, 'a')
        file.write(str(validation_loss) + '\n')
        file.close()

        print("Validation Loss: " + str(validation_loss))


history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=[SentenceInfo(validation_dataset)])

# outputs the time

dif_time = time.time() - start_time
               
file = open(time_file, 'w')
file.write(str(dif_time))
file.close()

print(dif_time)

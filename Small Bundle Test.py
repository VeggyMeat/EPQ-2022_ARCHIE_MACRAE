import numpy as np
import os
import shutil
import time

# disables tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

counter = 0

data_in = "/home/amri123/Desktop/Training Data"
data_to = "/home/amri123/Desktop/Training Data/Data"
labels_to = "/home/amri123/Desktop/Training Data/Labels"

start_time = time.time()
leading_zeros = 7
total_files = 104014
one_percent = total_files // 100

for root, dirs, files in os.walk("/home/amri123/Desktop/Training Data/train-clean-360/LibriSpeech/train-clean-360"):
    # makes sure its the appropraite depth, where the files are
    if root.count("/") == 9:
        # makes sure they come in numerical order
        files.sort()

        # loads the text files and the lines
        text_file = open(os.path.join(root, files[-1]), 'r')
        lines = text_file.readlines()
        text_file.close()
        for n, file in enumerate(files):
            if file.endswith(".flac"):
                # moves the file to the right spot
                shutil.copy(os.path.join(root, file), data_to)
                # renames the file
                os.rename(os.path.join(data_to, file), os.path.join(data_to, (leading_zeros - len(str(counter))) * '0' + str(counter) + '.flac'))

                # generates the label file
                label_file = open(os.path.join(labels_to, (leading_zeros - len(str(counter))) * '0' + str(counter) + '.txt'), 'w')
                label_file.write(' '.join(lines[n].split(' ')[1:]))
                label_file.close()
                counter += 1

            if counter % one_percent == 0:
                # prints an estimated time and percentage done message
                percent_done = counter // one_percent
                time_elapsed = time.time() - start_time
                print(str(percent_done) + " percent done")
                print("estimated " + str((time_elapsed / percent_done) * (100 - percent_done))[:7] + " seconds left")

print(counter)

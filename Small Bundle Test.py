import numpy as np
import os
import shutil
# disables tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

counter = 0

leading_zeros = 7
data_to = "/home/amri123/Desktop/Training Data/Data"
labels_to = "/home/amri123/Desktop/Training Data/Labels"
total_files = 252702

for root, dirs, files in os.walk("/home/amri123/Desktop/Training Data/train-clean-360/LibriSpeech/train-clean-360/192/128345"):
    if root.count("/") == 9:
        # makes sure they come in numerical order
        files.sort()
        text_file = open(os.path.join(root, files[-1]), 'r')
        lines = text_file.readlines()
        text_file.close()
        for n, file in enumerate(files):
            if file.endswith(".flac"):
                counter += 1
                # moves the file to the right spot
                shutil.copy(os.path.join(root, file), data_to)
                # renames the file
                os.rename(os.path.join(data_to, file), os.path.join(data_to, (leading_zeros - len(str(counter))) * '0' + str(counter) + '.flac'))

                # generates the label file
                label_file = open(os.path.join(labels_to, (leading_zeros - len(str(counter))) * '0' + str(counter) + '.txt'), 'w')
                label_file.write(' '.join(lines[n].split(' ')[1:]))
                label_file.close()

print(counter)

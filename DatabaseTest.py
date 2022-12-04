import numpy as np
import os
# disables tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

counter = 0

for root, dirs, files in os.walk("/home/amri123/Desktop/Training Data"):
    if root.count("/") == 9:
        # makes sure they come in numerical order
        files.sort()
        for file in files:
            if file.endswith(".flac"):
                counter += 1
                print(file)

print(counter)

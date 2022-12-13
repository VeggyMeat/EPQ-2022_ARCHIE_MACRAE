import os
# disables tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
import tensorflow_io as tfio
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

labels_dir = "/home/amri123/Desktop/Training Data/Labels/"


def character_check():
    # sets the starting variables and the number of files I have
    total_files = 252702
    chars = {}

    # loops through a count up to file number
    for count in range(total_files):
        # opens the file of that one
        label_file = open(os.path.join(labels_dir, (7 - len(str(count))) * '0' + str(count) + '.txt'), 'r')
        data = label_file.read()

        # goes through each character and adds that letter / increments that letter in the dictionary
        for char in data:
            if char in chars:
                chars[char] += 1
            else:
                chars[char] = 1
        
        label_file.close()
    
    return chars


# print(character_check())

# characters I need to accept
chars= "ABCDEFGHIJKLMNOPQRSTUVWXYZ' "

# dictionary of character to number
char_map = {chars[x]: x + 1 for x in range(len(chars))}


def data_to_num(data):
    # returns a list of the numbers associated to each character
    return [char_map[char] for char in data]


def label_change_num(file_name):
    # opens the file
    file = open(file_name, 'r')

    # reads all the data except the blank line character at the end
    data = file.read()[:-1]
    file.close()

    # translates it into a list of numbers
    numbers = data_to_num(data)
    numbers = [str(num) for num in numbers]

    # writes the data back as the numbers
    file = open(file_name, 'w')
    file.write(' '.join(numbers))
    file.close()
    print(' '.join(numbers))


label_change_num("/home/amri123/Desktop/Training Data/0000000.txt")

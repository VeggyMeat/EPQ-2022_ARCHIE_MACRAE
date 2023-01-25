import os
# disables tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import time

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
char_map[''] = 0

inv_map = {x + 1: chars[x] for x in range(len(chars))}
inv_map[0] = ''


def data_to_num(data):
    # returns a list of the numbers associated to each character
    return [char_map[char] for char in data]


def num_to_data(num):
    return [inv_map[char] for char in num]


def label_change_num(file_name, file_name_out):
    # opens the file
    file = open(file_name, 'r')

    # reads all the data except the blank line character at the end
    data = file.read()[:-1]
    file.close()

    # translates it into a list of numbers
    numbers = data_to_num(data)
    numbers = [str(num) for num in numbers]

    # writes the data back as the numbers
    file = open(file_name_out, 'w')
    file.write(' '.join(numbers))
    file.close()


def read_num_file(file_name):
    # opens the file
    file = tf.io.read_file(file_name)

    # reads in the data and makes it a list of integers
    data = tf.strings.split(file, ' ')
    data = tf.strings.to_number(data, out_type=tf.int32)

    return data


def labels_files_conversion(dir_in, dir_out, n):
    start_time = time.time()
    # repeats n times
    one_percent = n // 100
    for counter in range(n):
        # uses the label change num function with the directories with laeding 0 files
        label_change_num(os.path.join(dir_in, str(counter).zfill(7) + ".txt"), os.path.join(dir_out, str(counter).zfill(7) + ".txt"))

        if counter % one_percent == 0:
                # prints an estimated time and percentage done message
                if counter != 0:
                    percent_done = counter // one_percent
                    print(str(percent_done) + " percent done")
                    time_elapsed = time.time() - start_time
                    print("estimated " + str((time_elapsed / percent_done) * (100 - percent_done))[:7] + " seconds left", end='\n\n')

    print(str(time.time() - start_time) + " seconds taken")

# labels_files_conversion("/home/amri123/Desktop/Training Data/Labels", "/media/amri123/External SSD/Labels", 252702)

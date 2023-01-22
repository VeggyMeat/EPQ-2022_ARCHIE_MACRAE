import time
from tensorflow import keras
import tensorflow as tf
import os
import Label_Handling

'''print(tf.constant(Label_Handling.data_to_num("HELLO WORLD"), dtype="int64"))
print(keras.layers.StringLookup(vocabulary=[x for x in "abcdefghijklmnopqrstuvwxyz' "], oov_token="")(tf.strings.unicode_split("hello world", input_encoding="UTF-8")))

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
  tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8150)])
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus, 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)'''

# 13000
for x in range(10):
  name = '/media/amri123/External SSD/Labels/' + str(x).zfill(7) + '.txt'
  file = open(name, 'r')
  data = file.read()
  # numbers = data.split(' ')
  # numbers = [int(item) - 1 for item in numbers]
  file.close()

  print(data)
  
  # file = open(name, 'w')
  # file.write(' '.join([str(num) for num in numbers]))
  # file.close()

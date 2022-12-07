from tensorflow import keras
import tensorflow as tf

print(tf.constant([8, 5, 12, 12, 15, 27, 23, 15, 18, 12, 4], dtype="int64"))
print(keras.layers.StringLookup(vocabulary=[x for x in "abcdefghijklmnopqrstuvwxyz "], oov_token="")(tf.strings.unicode_split("hello world", input_encoding="UTF-8")))
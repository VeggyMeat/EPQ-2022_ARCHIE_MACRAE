import tensorflow as tf
from tensorflow import keras
import tensorflow_io as tfio

def to_spectogram(data_file_loc):
    data_data = tf.io.read_file(data_file_loc)

    audio = tfio.audio.decode_flac(data_data)
    audio = tf.squeeze(audio, axis=-1)

    audio = tf.cast(audio, tf.float32)

    spectrogram = tf.signal.stft(audio, frame_length=256, frame_step=160, fft_length=384)

    spectrogram = tf.abs(spectrogram)
    # spectrogram = tf.math.pow(spectrogram, 0.5)

    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    return spectrogram

print(to_spectogram("/home/amri123/Desktop/Training Data/Data/0000000.flac"))
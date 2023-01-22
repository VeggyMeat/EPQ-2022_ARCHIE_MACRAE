import Label_Handling
import Spectrogram
import tensorflow as tf
from CNN_RNN_Model import model_creation, CTCLoss

model = model_creation(193, len(Label_Handling.chars))

optimiser = tf.keras.optimizers.Adam()

batch_size = 12
total_files = 252702
epochs = 1
spectrogram_dir = "/media/amri123/External SSD/Spectrograms"
labels_dir = "/media/amri123/External SSD/Labels"

files_train = 12000
files_validate = 120

train_spectrograms = [os.path.join(spectrogram_dir, str(x).zfill(7) + '.npz') for x in range(files_train)]
validate_spectrograms = [os.path.join(spectrogram_dir, str(x).zfill(7) + 'npz') for x in range(files_train, files_train + files_validate)]

train_labels = [os.path.join(labels_dir, str(x).zfill(7) + '.txt') for x in range(files_train)]
validate_labels = [os.path.join(labels_dir, str(x).zfill(7) + 'txt') for x in range(files_train, files_train + files_validate)]

# model.summary()

num = 0

print("huh")

for epoch in range(epochs):
    epoch_loss_average = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for batch in range(files_train // batch_size):
        print("ok")
        labels = []
        predicteds = []

        for file in range(batch_size):
            label = Label_Handling.read_num_file(train_labels[num])
            labels.append(label)

            spectrogram = Spectrogram.read_spectrogram_file(train_spectrograms[num])

            predicted = model(spectrogram, training=True)
            predicteds.append(predicted)

            num += 1
        
        print(labels)
        print(predicteds)

        predicteds = tf.ragged.constant(predicteds, dtype=tf.int64)
        labels = tf.ragged.constant(labels, dtype=tf.int64)

        print(labels)
        print(predicteds)

        loss = CTCLoss(labels, predicteds)

        with tf.GradientTape() as tape:
            gradients = tape.gradient(loss, model.trainable_variables)
        
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))

        epoch_loss_average.update_state(loss)
        epoch_accuracy.update_state(labels, predicteds)

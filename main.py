import sys, os
import numpy as np
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Conv3D, MaxPooling3D
from utility_functions import opening_files
from create_partition import create_partition_and_labels
from DataGenerator import DataGenerator
from keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy


# get arguments
# _, generate_samples = sys.argv
root_path = os.getcwd()
sample_dir = "/".join([root_path, "samples"])

# create partition
partition, labels = create_partition_and_labels(sample_dir, 0.8, randomise=True)

# generators
# Parameters
params = {'dim': (28, 28, 28),
          'batch_size': 16,
          'n_channels': 1,
          'shuffle': True}

training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Input
model = Sequential()
model.add(Conv3D(64, kernel_size=(5, 5, 5), strides=(1, 1, 1), activation='sigmoid', padding="same",
                 input_shape=(28, 28, 28, 1)))
model.add(Conv3D(64, kernel_size=(5, 5, 5), strides=(1, 1, 1), activation='sigmoid', padding="same"))
model.add(Conv3D(64, kernel_size=(5, 5, 5), strides=(1, 1, 1), activation='sigmoid', padding="same"))
model.add(Conv3D(64, kernel_size=(5, 5, 5), strides=(1, 1, 1), activation='sigmoid', padding="same"))
model.add(Conv3D(64, kernel_size=(5, 5, 5), strides=(1, 1, 1), activation='sigmoid', padding="same"))
model.add(Conv3D(2, kernel_size=(5, 5, 5), strides=(1, 1, 1), activation='softmax', padding="same"))

weights = np.array([0.1, 0.9])

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=weighted_categorical_crossentropy(weights), metrics=['categorical_accuracy'])

for layer in model.layers:
    print(layer.name)
    print(layer.input_shape)
    print(layer.output_shape)

# train the model
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    epochs=20)

model.save('main-model.h5')

# show predictions


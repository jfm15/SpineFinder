import sys, os
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Conv3D, MaxPooling3D
from utility_functions import opening_files
from create_partition import create_partition_and_labels
from DataGenerator import DataGenerator


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
model.add(Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding="same",
                 input_shape=(None, None, None, 1)))
#model.add(Conv3D(64, kernel_size=(5, 5, 5), strides=(1, 1, 1), activation='relu', padding="same"))
#model.add(Conv3D(64, kernel_size=(5, 5, 5), strides=(1, 1, 1), activation='relu', padding="same"))
model.add(Conv3D(27, kernel_size=(5, 5, 5), strides=(1, 1, 1), activation='softmax', padding="same"))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

for layer in model.layers:
    print(layer.name)
    print(layer.input_shape)
    print(layer.output_shape)

# train the model
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    epochs=5)

model.save('main-model.h5')

# show predictions


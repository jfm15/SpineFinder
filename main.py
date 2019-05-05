import sys, os
from keras.models import Model
from keras.layers import Input, Flatten, Conv3D, MaxPooling3D
from utility_functions import opening_files
from create_partition import create_partition_and_labels
from DataGenerator import DataGenerator
from packages.UnetCNN.unet3d.model import unet_model_3d


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
model = unet_model_3d(input_shape=(1, 28, 28, 28),
                      n_labels=27,
                      depth=2,
                      pool_size=(2, 2, 2))

# train the model
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    epochs=1)

model.save('main-model.h5')

# show predictions


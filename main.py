import numpy as np
from create_partition import create_partition_and_labels
from data_generator import DataGenerator
from models.six_conv_multi_classes import six_conv_multi_classes
from models.six_conv_two_classes import six_conv_two_classes
from models.unet import unet

sample_dir = "/".join(["samples", "multi_class"])

# create partition
partition, labels = create_partition_and_labels(sample_dir, 0.8, randomise=True)

# generators
# Parameters
params = {'dim': (28, 28, 28),
          'samples_dir': sample_dir,
          'batch_size': 16,
          'n_channels': 1,
          'n_classes': 28,
          'shuffle': True}

training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

'''
model = unet(input_shape=(28, 28, 28),
             kernel_size=(3, 3, 3),
             weights=np.array([0.7, 0.93]),
             learning_rate=0.003)
'''

'''
model = six_conv_two_classes(input_shape=(28, 28, 28, 1),
                             kernel_size=(5, 5, 5),
                             weights=np.array([0.1, 0.9]))
'''

weights = np.ones(28) * 0.0384
weights[0] = 0.002
model = six_conv_multi_classes(input_shape=(28, 28, 28, 1),
                               kernel_size=(5, 5, 5),
                               weights=weights)

for layer in model.layers:
    print(layer.name)
    print(layer.input_shape)
    print(layer.output_shape)

# train the model
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    epochs=40)

model.save('main-model.h5')
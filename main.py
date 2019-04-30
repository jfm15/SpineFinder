import sys, os
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D
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
params = {'dim': (128, 128, 32),
          'batch_size': 16,
          'n_channels': 1,
          'shuffle': True}

training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Input
main_input = Input(shape=(128, 128, 32, 1))

x = Conv3D(64, kernel_size=(5, 5, 3), strides=(2, 2, 2), activation='relu', padding="same",
                 input_shape=(None, None, None, 1))(main_input)

x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

x = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding="same")(x)

x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

x = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding="same")(x)

x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

x = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding="same")(x)

x = Conv3D(1024, kernel_size=(8, 8, 2), strides=(1, 1, 1), activation='relu')(x)

position_predictions = Conv3D(3, kernel_size=(1, 1, 1), strides=(1, 1, 1), name="position_predictions")(x)

label_predictions = Conv3D(27, kernel_size=(1, 1, 1), strides=(1, 1, 1), name="label_predictions")(x)
# output_shape=(1, 1, 1, 3)

"""
for layer in model.layers:
    print(layer.name)
    print(layer.input_shape)
    print(layer.output_shape)
"""

model = Model(inputs=main_input, outputs=[position_predictions, label_predictions])

model.compile(optimizer='adam',
              loss={"position_predictions": 'mean_absolute_error',
                    "label_predictions": "categorical_crossentropy"},
              metrics=['accuracy'])

# train the model
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    epochs=6)

model.save('main-model.h5')

# show predictions


import numpy as np
import keras_metrics as km
import keras.metrics as metrics
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D


def six_conv_slices(kernel_size):
    # Input
    model = Sequential()
    model.add(Conv2D(64, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding="same",
                     input_shape=(None, None, 1)))
    model.add(Conv2D(64, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding="same"))
    model.add(Conv2D(64, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding="same"))
    model.add(Conv2D(64, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding="same"))
    model.add(Conv2D(64, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding="same"))
    model.add(Conv2D(1, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding="same"))

    # define optimizer
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss="mean_absolute_error", metrics=["mean_absolute_error"])

    return model
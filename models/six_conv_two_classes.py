import numpy as np
import keras_metrics as km
import keras.metrics as metrics
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv3D
from losses_and_metrics.keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy


def six_conv_two_classes(input_shape, kernel_size, weights):
    # Input
    model = Sequential()
    model.add(Conv3D(16, kernel_size=kernel_size, strides=(1, 1, 1), activation='relu', padding="same",
                     input_shape=input_shape))
    model.add(Conv3D(16, kernel_size=kernel_size, strides=(1, 1, 1), activation='relu', padding="same"))
    model.add(Conv3D(16, kernel_size=kernel_size, strides=(1, 1, 1), activation='relu', padding="same"))
    model.add(Conv3D(2, kernel_size=kernel_size, strides=(1, 1, 1), activation='softmax', padding="same"))

    # define optimizer
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # define loss function
    loss_function = weighted_categorical_crossentropy(weights)

    # define metrics
    recall_background = km.binary_recall(label=0)
    recall_vertebrae = km.binary_recall(label=1)
    cat_accuracy = metrics.categorical_accuracy

    model.compile(optimizer=sgd, loss=loss_function, metrics=[recall_background, recall_vertebrae, cat_accuracy])

    return model
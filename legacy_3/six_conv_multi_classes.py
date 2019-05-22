import numpy as np
import keras_metrics as km
import keras.metrics as metrics
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv3D
from losses_and_metrics.keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy


def six_conv_multi_classes(input_shape, kernel_size, classes, weights, learning_rate=0.1):
    # Input
    model = Sequential()
    model.add(Conv3D(64, kernel_size=kernel_size, strides=(1, 1, 1), activation='sigmoid', padding="same",
                     input_shape=input_shape))
    model.add(Conv3D(64, kernel_size=kernel_size, strides=(1, 1, 1), activation='sigmoid', padding="same"))
    model.add(Conv3D(64, kernel_size=kernel_size, strides=(1, 1, 1), activation='sigmoid', padding="same"))
    model.add(Conv3D(64, kernel_size=kernel_size, strides=(1, 1, 1), activation='sigmoid', padding="same"))
    model.add(Conv3D(64, kernel_size=kernel_size, strides=(1, 1, 1), activation='sigmoid', padding="same"))
    model.add(Conv3D(classes, kernel_size=kernel_size, strides=(1, 1, 1), activation='softmax', padding="same"))

    # define optimizer
    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

    # define loss function
    loss_function = weighted_categorical_crossentropy(weights)

    # define metrics
    list_of_metrics = []
    for i in range(0, classes):
        list_of_metrics.append(km.binary_recall(label=i))
        
    list_of_metrics.append(metrics.categorical_accuracy)

    model.compile(optimizer=sgd, loss=loss_function, metrics=list_of_metrics)

    return model

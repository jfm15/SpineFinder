import numpy as np
import keras_metrics as km
import keras.metrics as metrics
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv3D, BatchNormalization, Activation
from losses_and_metrics.keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy
from losses_and_metrics.dsc import dice_coef_label


def simple_detection(input_shape, filters, kernel_size, weights, learning_rate):
    # Input
    model = Sequential()
    model.add(Conv3D(filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same",
                     input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.1))
    model.add(Activation("relu"))
    model.add(Conv3D(filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same"))
    model.add(BatchNormalization(momentum=0.1))
    model.add(Activation("relu"))
    model.add(Conv3D(filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same"))
    model.add(BatchNormalization(momentum=0.1))
    model.add(Activation("relu"))
    model.add(Conv3D(filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same"))
    model.add(BatchNormalization(momentum=0.1))
    model.add(Activation("relu"))
    model.add(Conv3D(2, kernel_size=kernel_size, strides=(1, 1, 1), activation='softmax', padding="same"))

    # define optimizer
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # NOTE: if any of the below parameters change then change the description file
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)

    # define loss function
    loss_function = weighted_categorical_crossentropy(weights)

    # define metrics
    dsc = dice_coef_label(label=1)
    recall_background = km.binary_recall(label=0)
    recall_vertebrae = km.binary_recall(label=1)
    cat_accuracy = metrics.categorical_accuracy

    model.compile(optimizer=adam, loss=loss_function, metrics=[dsc, recall_background, recall_vertebrae, cat_accuracy])

    return model
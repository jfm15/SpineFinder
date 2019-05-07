import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv3D
from losses_and_metrics.keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy


def six_conv_two_classes():
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

    return model
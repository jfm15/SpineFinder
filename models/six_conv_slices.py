import numpy as np
import keras_metrics as km
import keras.metrics as metrics
from keras import optimizers
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, concatenate


def six_conv_slices(kernel_size):
    # Input
    main_input = Input(shape=(None, None, 1))
    x = Conv2D(64, kernel_size=kernel_size, strides=(1, 1), activation='sigmoid', padding="same")(main_input)
    x = Conv2D(256, kernel_size=kernel_size, strides=(1, 1), activation='sigmoid', padding="same")(x)
    x = Conv2D(256, kernel_size=kernel_size, strides=(1, 1), activation='sigmoid', padding="same")(x)
    x = Conv2D(256, kernel_size=kernel_size, strides=(1, 1), activation='sigmoid', padding="same")(x)
    x = Conv2D(256, kernel_size=kernel_size, strides=(1, 1), activation='sigmoid', padding="same")(x)
    x = Conv2D(256, kernel_size=kernel_size, strides=(1, 1), activation='sigmoid', padding="same")(x)
    x = Conv2D(256, kernel_size=(1, 100), strides=(1, 1), activation='sigmoid', padding="same")(x)
    x = Conv2D(256, kernel_size=(24, 1), strides=(1, 1), activation='sigmoid', padding="same")(x)
    main_output = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same")(x)

    model = Model(inputs=main_input, outputs=main_output)

    # define optimizer
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss=cool_loss, metrics=["mean_absolute_error", "mean_squared_error"])

    return model


def cool_loss(y_true, y_pred):
    dont_cares = K.minimum(1.0, y_true)
    return K.sum(K.abs(y_pred - y_true) * dont_cares) / K.sum(dont_cares)


def unet_slices(kernel_size):

    main_input = Input(shape=(None, None, 1))

    step_down_1 = Conv2D(64, kernel_size=kernel_size, strides=(1, 1), padding="same",
                         activation='sigmoid')(main_input)
    step_down_1 = Conv2D(64, kernel_size=kernel_size, strides=(1, 1), padding="same",
                         activation='sigmoid')(step_down_1)

    step_down_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(step_down_1)
    step_down_2 = Conv2D(128, kernel_size=kernel_size, strides=(1, 1), padding="same",
                         activation='sigmoid')(step_down_2)
    step_down_2 = Conv2D(128, kernel_size=kernel_size, strides=(1, 1), padding="same",
                         activation='sigmoid')(step_down_2)

    floor = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(step_down_2)
    floor = Conv2D(256, kernel_size=kernel_size, strides=(1, 1), padding="same",
                   activation='sigmoid')(floor)
    floor = Conv2D(256, kernel_size=kernel_size, strides=(1, 1), padding="same",
                   activation='sigmoid')(floor)

    step_up_2 = UpSampling2D(size=(2, 2))(floor)
    step_up_2 = concatenate([step_down_2, step_up_2], axis=-1)
    step_up_2 = Conv2D(128, kernel_size=kernel_size, strides=(1, 1), padding="same",
                       activation='sigmoid')(step_up_2)
    step_up_2 = Conv2D(128, kernel_size=kernel_size, strides=(1, 1), padding="same",
                       activation='sigmoid')(step_up_2)

    step_up_1 = UpSampling2D(size=(2, 2))(step_up_2)
    step_up_1 = concatenate([step_down_1, step_up_1], axis=-1)
    step_up_1 = Conv2D(64, kernel_size=kernel_size, strides=(1, 1), padding="same",
                       activation='sigmoid')(step_up_1)
    step_up_1 = Conv2D(64, kernel_size=kernel_size, strides=(1, 1), padding="same",
                       activation='sigmoid')(step_up_1)

    main_output = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same",
                         activation='relu')(step_up_1)

    model = Model(inputs=main_input, outputs=main_output)

    # define optimizer
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss="mean_squared_error", metrics=["mean_absolute_error", "mean_squared_error"])

    return model
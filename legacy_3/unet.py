import numpy as np
import keras_metrics as km
import keras.metrics as metrics
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from losses_and_metrics.keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy


def unet(input_shape, kernel_size, weights, learning_rate=0.05):

    main_input = Input(shape=(28, 28, 28, 1))

    # 28^2
    step_down_1 = Conv3D(64, kernel_size=kernel_size, strides=(1, 1, 1), padding="same",
                    activation='sigmoid')(main_input)
    step_down_1 = Conv3D(64, kernel_size=kernel_size, strides=(1, 1, 1), padding="same",
                    activation='sigmoid')(step_down_1)

    # 14^2
    step_down_2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(step_down_1)
    step_down_2 = Conv3D(128, kernel_size=kernel_size, strides=(1, 1, 1), padding="same",
                    activation='sigmoid')(step_down_2)
    step_down_2 = Conv3D(128, kernel_size=kernel_size, strides=(1, 1, 1), padding="same",
                    activation='sigmoid')(step_down_2)

    # 7^2
    floor = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(step_down_2)
    floor = Conv3D(256, kernel_size=kernel_size, strides=(1, 1, 1), padding="same",
                    activation='sigmoid')(floor)
    floor = Conv3D(256, kernel_size=kernel_size, strides=(1, 1, 1), padding="same",
                    activation='sigmoid')(floor)

    # 14^2
    step_up_2 = UpSampling3D(size=(2, 2, 2))(floor)
    step_up_2 = concatenate([step_down_2, step_up_2], axis=-1)
    step_up_2 = Conv3D(128, kernel_size=kernel_size, strides=(1, 1, 1), padding="same",
                         activation='sigmoid')(step_up_2)
    step_up_2 = Conv3D(128, kernel_size=kernel_size, strides=(1, 1, 1), padding="same",
                         activation='sigmoid')(step_up_2)

    # 28^2
    step_up_1 = UpSampling3D(size=(2, 2, 2))(step_up_2)
    step_up_1 = concatenate([step_down_1, step_up_1], axis=-1)
    step_up_1 = Conv3D(64, kernel_size=kernel_size, strides=(1, 1, 1), padding="same",
                       activation='sigmoid')(step_up_1)
    step_up_1 = Conv3D(64, kernel_size=kernel_size, strides=(1, 1, 1), padding="same",
                       activation='sigmoid')(step_up_1)

    main_output = Conv3D(2, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same",
                       activation='softmax')(step_up_1)

    model = Model(inputs=main_input, outputs=main_output)

    # define optimizer
    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

    # define loss function
    loss_function = weighted_categorical_crossentropy(weights)

    # define metrics
    recall_background = km.binary_recall(label=0)
    recall_vertebrae = km.binary_recall(label=1)
    cat_accuracy = metrics.categorical_accuracy

    model.compile(optimizer=sgd, loss=loss_function, metrics=[recall_background, recall_vertebrae, cat_accuracy])

    return model
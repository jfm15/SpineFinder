import numpy as np
import keras_metrics as km
import keras.metrics as metrics
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Conv3D, BatchNormalization, Activation, Input, MaxPooling3D, concatenate, UpSampling3D, Cropping3D
from losses_and_metrics.keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy
from losses_and_metrics.dsc import dice_coef_label


def simple_detection(filters, kernel_size, weights, learning_rate):
    # Input
    model = Sequential()
    model.add(Conv3D(filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same",
                     input_shape=(None, None, None, 1)))
    # model.add(BatchNormalization(momentum=0.1))
    model.add(Activation("relu"))
    model.add(Conv3D(filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same"))
    # model.add(BatchNormalization(momentum=0.1))
    model.add(Activation("relu"))
    model.add(Conv3D(filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same"))
    # model.add(BatchNormalization(momentum=0.1))
    model.add(Activation("relu"))
    model.add(Conv3D(filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same"))
    # model.add(BatchNormalization(momentum=0.1))
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


def detection_unet(filters, kernel_size, weights, learning_rate):

    # Input
    main_input = Input(shape=(None, None, None, 1))

    # 64 x 64 x 80
    step_down_1 = Conv3D(filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(main_input)
    step_down_1 = BatchNormalization(momentum=0.1)(step_down_1)
    step_down_1 = Activation("relu")(step_down_1)
    step_down_1 = Conv3D(filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(step_down_1)
    step_down_1 = BatchNormalization(momentum=0.1)(step_down_1)
    step_down_1 = Activation("relu")(step_down_1)

    # 32 x 32 x 40
    step_down_2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(step_down_1)
    step_down_2 = Conv3D(2 * filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(step_down_2)
    step_down_2 = BatchNormalization(momentum=0.1)(step_down_2)
    step_down_2 = Activation("relu")(step_down_2)
    step_down_2 = Conv3D(2 * filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(step_down_2)
    step_down_2 = BatchNormalization(momentum=0.1)(step_down_2)
    step_down_2 = Activation("relu")(step_down_2)

    # 16 x 16 x 20
    step_down_3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(step_down_2)
    step_down_3 = Conv3D(4 * filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(step_down_3)
    step_down_3 = BatchNormalization(momentum=0.1)(step_down_3)
    step_down_3 = Activation("relu")(step_down_3)
    step_down_3 = Conv3D(4 * filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(step_down_3)
    step_down_3 = BatchNormalization(momentum=0.1)(step_down_3)
    step_down_3 = Activation("relu")(step_down_3)

    # 8 x 8 x 10
    step_down_4 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(step_down_3)
    step_down_4 = Conv3D(8 * filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(step_down_4)
    step_down_4 = BatchNormalization(momentum=0.1)(step_down_4)
    step_down_4 = Activation("relu")(step_down_4)
    step_down_4 = Conv3D(8 * filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(step_down_4)
    step_down_4 = BatchNormalization(momentum=0.1)(step_down_4)
    step_down_4 = Activation("relu")(step_down_4)

    # 4 x 4 x 5
    floor = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(step_down_4)
    floor = Conv3D(16 * filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(floor)
    floor = BatchNormalization(momentum=0.1)(floor)
    floor = Activation("relu")(floor)
    floor = Conv3D(16 * filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(floor)
    floor = BatchNormalization(momentum=0.1)(floor)
    floor = Activation("relu")(floor)

    # 8 x 8 x 10
    step_up_4 = UpSampling3D(size=(2, 2, 2))(floor)
    step_up_4 = concatenate([step_down_4, step_up_4], axis=-1)
    step_up_4 = Conv3D(8 * filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(step_up_4)
    step_up_4 = BatchNormalization(momentum=0.1)(step_up_4)
    step_up_4 = Activation("relu")(step_up_4)
    step_up_4 = Conv3D(8 * filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(step_up_4)
    step_up_4 = BatchNormalization(momentum=0.1)(step_up_4)
    step_up_4 = Activation("relu")(step_up_4)

    # 16 x 16 x 20
    step_up_3 = UpSampling3D(size=(2, 2, 2))(step_up_4)
    step_up_3 = concatenate([step_down_3, step_up_3], axis=-1)
    step_up_3 = Conv3D(4 * filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(step_up_3)
    step_up_3 = BatchNormalization(momentum=0.1)(step_up_3)
    step_up_3 = Activation("relu")(step_up_3)
    step_up_3 = Conv3D(4 * filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(step_up_3)
    step_up_3 = BatchNormalization(momentum=0.1)(step_up_3)
    step_up_3 = Activation("relu")(step_up_3)

    # 32 x 32 x 40
    step_up_2 = UpSampling3D(size=(2, 2, 2))(step_up_3)
    step_up_2 = concatenate([step_down_2, step_up_2], axis=-1)
    step_up_2 = Conv3D(2 * filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(step_up_2)
    step_up_2 = BatchNormalization(momentum=0.1)(step_up_2)
    step_up_2 = Activation("relu")(step_up_2)
    step_up_2 = Conv3D(2 * filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(step_up_2)
    step_up_2 = BatchNormalization(momentum=0.1)(step_up_2)
    step_up_2 = Activation("relu")(step_up_2)

    # 64 x 64 x 80
    step_up_1 = UpSampling3D(size=(2, 2, 2))(step_up_2)
    step_up_1 = concatenate([step_down_1, step_up_1], axis=-1)
    step_up_1 = Conv3D(filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(step_up_1)
    step_up_1 = BatchNormalization(momentum=0.1)(step_up_1)
    step_up_1 = Activation("relu")(step_up_1)
    step_up_1 = Conv3D(filters, kernel_size=kernel_size, strides=(1, 1, 1), padding="same")(step_up_1)
    step_up_1 = BatchNormalization(momentum=0.1)(step_up_1)
    step_up_1 = Activation("relu")(step_up_1)

    main_output = Conv3D(2, kernel_size=kernel_size, strides=(1, 1, 1), padding="same",
                         activation='softmax')(step_up_1)

    model = Model(inputs=main_input, outputs=main_output)

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


def detection_unet_no_padding(filters, kernel_size, weights, learning_rate):
    # Input
    main_input = Input(shape=(None, None, None, 1))

    # 68 x 68 x 84
    step_down_1 = Conv3D(filters, kernel_size=kernel_size, strides=(1, 1, 1))(main_input)
    step_down_1 = BatchNormalization(momentum=0.1)(step_down_1)
    step_down_1 = Activation("relu")(step_down_1)
    # 66 x 66 x 82
    step_down_1 = Conv3D(filters, kernel_size=kernel_size, strides=(1, 1, 1))(step_down_1)
    step_down_1 = BatchNormalization(momentum=0.1)(step_down_1)
    step_down_1 = Activation("relu")(step_down_1)

    # 64 x 64 x 80
    step_down_2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(step_down_1)
    # 32 x 32 x 40
    step_down_2 = Conv3D(2 * filters, kernel_size=kernel_size, strides=(1, 1, 1))(step_down_2)
    step_down_2 = BatchNormalization(momentum=0.1)(step_down_2)
    step_down_2 = Activation("relu")(step_down_2)
    # 30 x 30 x 38
    step_down_2 = Conv3D(2 * filters, kernel_size=kernel_size, strides=(1, 1, 1))(step_down_2)
    step_down_2 = BatchNormalization(momentum=0.1)(step_down_2)
    step_down_2 = Activation("relu")(step_down_2)

    # 28 x 28 x 36
    floor = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(step_down_2)
    # 14 x 14 x 18
    floor = Conv3D(4 * filters, kernel_size=kernel_size, strides=(1, 1, 1))(floor)
    floor = BatchNormalization(momentum=0.1)(floor)
    floor = Activation("relu")(floor)
    # 12 x 12 x 16
    floor = Conv3D(4 * filters, kernel_size=kernel_size, strides=(1, 1, 1))(floor)
    floor = BatchNormalization(momentum=0.1)(floor)
    floor = Activation("relu")(floor)

    # 10 x 10 x 14
    step_up_2 = UpSampling3D(size=(2, 2, 2))(floor)
    # 20 x 20 x 28
    cropped_step_down_2 = Cropping3D(cropping=((4, 4), (4, 4), (4, 4)))(step_down_2)
    step_up_2 = concatenate([cropped_step_down_2, step_up_2], axis=-1)
    step_up_2 = Conv3D(2 * filters, kernel_size=kernel_size, strides=(1, 1, 1))(step_up_2)
    step_up_2 = BatchNormalization(momentum=0.1)(step_up_2)
    step_up_2 = Activation("relu")(step_up_2)
    # 18 x 18 x 26
    step_up_2 = Conv3D(2 * filters, kernel_size=kernel_size, strides=(1, 1, 1))(step_up_2)
    step_up_2 = BatchNormalization(momentum=0.1)(step_up_2)
    step_up_2 = Activation("relu")(step_up_2)

    # 16 x 16 x 24
    step_up_1 = UpSampling3D(size=(2, 2, 2))(step_up_2)
    # 32 x 32 x 48
    cropped_step_down_1 = Cropping3D(cropping=((16, 16), (16, 16), (16, 16)))(step_down_1)
    step_up_1 = concatenate([cropped_step_down_1, step_up_1], axis=-1)
    step_up_1 = Conv3D(filters, kernel_size=kernel_size, strides=(1, 1, 1))(step_up_1)
    step_up_1 = BatchNormalization(momentum=0.1)(step_up_1)
    step_up_1 = Activation("relu")(step_up_1)
    # 30 x 30 x 46
    step_up_1 = Conv3D(filters, kernel_size=kernel_size, strides=(1, 1, 1))(step_up_1)
    step_up_1 = BatchNormalization(momentum=0.1)(step_up_1)
    step_up_1 = Activation("relu")(step_up_1)

    # 28 x 28 x 44
    main_output = Conv3D(2, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                         activation='softmax')(step_up_1)

    model = Model(inputs=main_input, outputs=main_output)

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

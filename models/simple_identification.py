from keras import optimizers
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, concatenate, Activation, BatchNormalization


def simple_identification(input_shape, kernel_size, filters, learning_rate):

    model = Sequential()
    model.add(Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding="same",
                     input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.1))
    model.add(Activation("relu"))
    model.add(Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding="same"))
    model.add(BatchNormalization(momentum=0.1))
    model.add(Activation("relu"))
    model.add(Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding="same"))
    model.add(BatchNormalization(momentum=0.1))
    model.add(Activation("relu"))
    model.add(Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding="same"))
    model.add(BatchNormalization(momentum=0.1))
    model.add(Activation("relu"))
    model.add(Conv2D(1, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding="same"))

    # NOTE: if any of the below parameters change then change the description file
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)
    model.compile(optimizer=adam, loss=ignore_background_loss, metrics=[vertebrae_classification_rate])

    return model

'''
def six_conv_slices(kernel_size):
    # Input
    main_input = Input(shape=(None, None, 1))
    x = Conv2D(64, kernel_size=kernel_size, strides=(1, 1), activation='sigmoid', padding="same")(main_input)
    x = Conv2D(256, kernel_size=kernel_size, strides=(1, 1), activation='sigmoid', padding="same")(x)
    x = Conv2D(256, kernel_size=kernel_size, strides=(1, 1), activation='sigmoid', padding="same")(x)
    x = Conv2D(256, kernel_size=kernel_size, strides=(1, 1), activation='sigmoid', padding="same")(x)
    x = Conv2D(256, kernel_size=kernel_size, strides=(1, 1), activation='sigmoid', padding="same")(x)
    x = Conv2D(256, kernel_size=kernel_size, strides=(1, 1), activation='sigmoid', padding="same")(x)
    branch_1 = Conv2D(256, kernel_size=kernel_size, strides=(1, 1), activation='sigmoid', padding="same")(x)
    branch_1 = Conv2D(256, kernel_size=kernel_size, strides=(1, 1), activation='sigmoid', padding="same")(branch_1)
    branch_2 = Conv2D(256, kernel_size=(1, 100), strides=(1, 1), activation='sigmoid', padding="same")(x)
    branch_2 = Conv2D(256, kernel_size=(24, 1), strides=(1, 1), activation='sigmoid', padding="same")(branch_2)
    x = concatenate([branch_1, branch_2], axis=-1)
    main_output = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same")(x)

    model = Model(inputs=main_input, outputs=main_output)

    # define optimizer
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss=ignore_background_loss, metrics=["mean_absolute_error", "mean_squared_error"])

    return model
'''


def ignore_background_loss(y_true, y_pred):
    # y_true = K.maximum(y_true, K.epsilon())
    dont_cares = K.minimum(1.0, y_true)
    return K.sum(K.abs(y_pred - y_true) * dont_cares) / K.sum(dont_cares)


def vertebrae_classification_rate(y_true, y_pred):
    # y_true = K.maximum(y_true, K.epsilon())
    dont_cares = K.minimum(1.0, y_true)
    return K.sum(K.cast(K.equal(K.round(y_pred), y_true), 'float32') * dont_cares) / K.sum(dont_cares)


def unet_slices(input_shape, kernel_size, filters, learning_rate):

    main_input = Input(shape=(None, None, 1))

    # 80 x 320
    step_down_1 = Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(main_input)
    step_down_1 = BatchNormalization()(step_down_1)
    step_down_1 = Activation("relu")(step_down_1)
    step_down_1 = Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(step_down_1)
    step_down_1 = BatchNormalization()(step_down_1)
    step_down_1 = Activation("relu")(step_down_1)

    # 40 x 160
    step_down_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(step_down_1)
    step_down_2 = Conv2D(2 * filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(step_down_2)
    step_down_2 = BatchNormalization()(step_down_2)
    step_down_2 = Activation("relu")(step_down_2)
    step_down_2 = Conv2D(2 * filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(step_down_2)
    step_down_2 = BatchNormalization()(step_down_2)
    step_down_2 = Activation("relu")(step_down_2)

    # 20 x 80
    step_down_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(step_down_2)
    step_down_3 = Conv2D(4 * filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(step_down_3)
    step_down_3 = BatchNormalization()(step_down_3)
    step_down_3 = Activation("relu")(step_down_3)
    step_down_3 = Conv2D(4 * filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(step_down_3)
    step_down_3 = BatchNormalization()(step_down_3)
    step_down_3 = Activation("relu")(step_down_3)

    # 10 x 40
    step_down_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(step_down_3)
    step_down_4 = Conv2D(8 * filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(step_down_4)
    step_down_4 = BatchNormalization()(step_down_4)
    step_down_4 = Activation("relu")(step_down_4)
    step_down_4 = Conv2D(8 * filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(step_down_4)
    step_down_4 = BatchNormalization()(step_down_4)
    step_down_4 = Activation("relu")(step_down_4)

    # 5 x 20
    floor = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(step_down_4)
    floor = Conv2D(16 * filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(floor)
    floor = BatchNormalization()(floor)
    floor = Activation("relu")(floor)
    floor = Conv2D(16 * filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(floor)
    floor = BatchNormalization()(floor)
    floor = Activation("relu")(floor)

    # 10 x 40
    step_up_4 = UpSampling2D(size=(2, 2))(floor)
    step_up_4 = concatenate([step_down_4, step_up_4], axis=-1)
    step_up_4 = Conv2D(8 * filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(step_up_4)
    step_up_4 = BatchNormalization()(step_up_4)
    step_up_4 = Activation("relu")(step_up_4)
    step_up_4 = Conv2D(8 * filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(step_up_4)
    step_up_4 = BatchNormalization()(step_up_4)
    step_up_4 = Activation("relu")(step_up_4)

    # 20 x 80
    step_up_3 = UpSampling2D(size=(2, 2))(step_up_4)
    step_up_3 = concatenate([step_down_3, step_up_3], axis=-1)
    step_up_3 = Conv2D(4 * filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(step_up_3)
    step_up_3 = BatchNormalization()(step_up_3)
    step_up_3 = Activation("relu")(step_up_3)
    step_up_3 = Conv2D(4 * filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(step_up_3)
    step_up_3 = BatchNormalization()(step_up_3)
    step_up_3 = Activation("relu")(step_up_3)

    # 40 x 160
    step_up_2 = UpSampling2D(size=(2, 2))(step_up_3)
    step_up_2 = concatenate([step_down_2, step_up_2], axis=-1)
    step_up_2 = Conv2D(2 * filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(step_up_2)
    step_up_2 = BatchNormalization()(step_up_2)
    step_up_2 = Activation("relu")(step_up_2)
    step_up_2 = Conv2D(2 * filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(step_up_2)
    step_up_2 = BatchNormalization()(step_up_2)
    step_up_2 = Activation("relu")(step_up_2)

    # 80 x 320
    step_up_1 = UpSampling2D(size=(2, 2))(step_up_2)
    step_up_1 = concatenate([step_down_1, step_up_1], axis=-1)
    step_up_1 = Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(step_up_1)
    step_up_1 = BatchNormalization()(step_up_1)
    step_up_1 = Activation("relu")(step_up_1)
    step_up_1 = Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(step_up_1)
    step_up_1 = BatchNormalization()(step_up_1)
    step_up_1 = Activation("relu")(step_up_1)

    main_output = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same",
                         activation='relu')(step_up_1)

    model = Model(inputs=main_input, outputs=main_output)

    # define optimizer
    # sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9)
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-4)
    model.compile(optimizer=adam, loss=ignore_background_loss, metrics=[vertebrae_classification_rate])

    return model
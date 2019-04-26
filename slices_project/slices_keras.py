import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

import read_data

base_directory = '/homes/jfm15/SpineFinder/'

X, Y = read_data.get_slices()

print("data loaded")

training_examples = 4000
testing_examples = X.shape[0] - training_examples

X_train = X[:training_examples]
Y_train = Y[:training_examples]

X_test = X[training_examples:]
Y_test = Y[training_examples:]

# reshape
X_train = X_train.reshape(training_examples, 320, 40, 1) / 255.0
X_test = X_test.reshape(testing_examples, 320, 40, 1)

Y_train = Y_train.reshape(training_examples, 320, 40, 1) / 255.0
Y_test = Y_test.reshape(testing_examples, 320, 40, 1)

print(X_train.shape, X_test.shape)


import matplotlib.pyplot as plt


# create model
model = Sequential()

# add model layers
model.add(Conv2D(64, kernel_size=5, activation='relu', input_shape=(320, 40, 1), padding='same'))
#model.add(Conv2D(64, kernel_size=5, activation='relu', padding='same'))
#model.add(Conv2D(64, kernel_size=5, activation='relu', padding='same'))
model.add(Conv2D(1, kernel_size=1, activation='sigmoid', padding='same'))

import keras.backend as K


def weighted_binary_crossentropy(y_true, y_pred):

    one_weight = 0.9
    zero_weight = 0.1

    # Original binary crossentropy (see losses.py):
    # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

    # Calculate the binary crossentropy
    b_ce = K.binary_crossentropy(y_true, y_pred)

    # Apply the weights
    weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
    weighted_b_ce = weight_vector * b_ce

    # Return the mean error
    return K.mean(weighted_b_ce)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3)

p1 = plt.figure(1)
plt.imshow(Y_test[0].reshape(320, 40).T, aspect=8)
p1.show()

p2 = plt.figure(2)
plt.imshow(model.predict(X_test[:1]).reshape(320, 40).T, aspect=8)
p2.show()

input()

import sys
import os
from keras.models import Sequential
from keras.layers import Dense, Conv3D, Flatten, MaxPooling3D
from utility_functions import opening_files

# get arguments
_, predict = sys.argv
root_path = os.getcwd()

# get datasets
dataset = []

# get samples with training labels
X_train = []
y_train = []
X_test = []
y_test = []

# train model
model = Sequential()
# input_shape=(1, 32, 112, 96)
model.add(Conv3D(64, kernel_size=(3, 5, 5), strides=(2, 2, 2), activation='relu', padding="same"))
# input_shape=(64, 32, 112, 96)
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
# input_shape=(64, 16, 56, 48)
model.add(Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding="same"))
# input_shape=(64, 16, 56, 48)
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
# input_shape=(64, 8, 28, 24)
model.add(Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding="same"))
# input_shape=(64, 8, 28, 24)
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
# input_shape=(64, 4, 14, 12)
model.add(Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding="same"))
# input_shape=(64, 4, 14, 12)
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
# input_shape=(64, 2, 7, 6)
model.add(Conv3D(1024, kernel_size=(2, 7, 6), strides=(1, 1, 1), activation='relu', padding="same"))
# input_shape=(1024, 1, 1, 1)
model.add(Conv3D(3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same"))
# output_shape=(3, 1, 1, 1)

model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

# train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

# show predictions


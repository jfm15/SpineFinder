# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):

    def __init__(self, ids_in_set, labels, batch_size=32, dim=(32, 32, 32), n_channels=1
                 , shuffle=True):

        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.ids_in_set = ids_in_set
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):

        return int(np.floor(len(self.ids_in_set) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_lower = index * self.batch_size
        batch_upper = batch_lower + self.batch_size
        indexes = self.indexes[batch_lower:batch_upper]

        # Get selected ids
        ids_in_set_temp = [self.ids_in_set[k] for k in indexes]

        # Generate data from ids
        X, y = self.__data_generation(ids_in_set_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        no_of_ids = len(self.ids_in_set)
        self.indexes = np.arange(no_of_ids)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids_in_set_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 1, *self.dim))
        y = np.empty((self.batch_size, 1, *self.dim), dtype=int)

        # Generate data
        for i, ID in enumerate(ids_in_set_temp):
            # Store sample
            X[i, ] = np.load('samples/' + ID + '-sample.npy').reshape(1, 128, 128, 32)

            # Store values
            label_id = self.labels[ID]
            y[i, ] = np.load('samples/' + label_id + '.npy').reshape(1, 128, 128, 32)

        return X, y
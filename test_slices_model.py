import numpy as np
from keras.models import load_model

sample = np.load('samples/slices/2684937-4-sample.npy')

model = load_model('model_files/slices_model.h5')

i_padding = 4 - sample.shape[0] % 4
j_padding = 4 - sample.shape[1] % 4
sample = np.pad(sample, ((0, i_padding), (0, j_padding)), "edge")

prediction = model.predict(sample.reshape(1, *sample.shape, 1))

print(np.unique(prediction))
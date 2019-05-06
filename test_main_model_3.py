import numpy as np
from keras.models import load_model
import keras.losses
from keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy

path = 'samples/2551924-34'
sample = np.load(path + '-sample.npy')
sample_labelling = np.load(path + '-labelling.npy')

weights = np.ones(27)
weights[0] = 0.001
weights /= np.sum(weights)

model = load_model('main-model.h5', custom_objects={'loss': weighted_categorical_crossentropy(weights)})

result = model.predict(sample.reshape(1, 28, 28, 28, 1))
result = result.reshape(28, 28, 28, 27)
print(np.around(result[5, 12, 21], decimals=2))
i, j, k = np.where(sample_labelling > 0)
print(i, j, k)
print(sample_labelling[21, 12, 5])
decat_result = np.argmax(result, axis=3)

print(np.bincount(sample_labelling.reshape(-1).astype(int)))
print(np.bincount(decat_result.reshape(-1).astype(int)))

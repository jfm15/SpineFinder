import numpy as np
from keras.models import load_model
from losses_and_metrics.keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

path = 'samples/2551924-34'
sample = np.load(path + '-sample.npy')
sample_labelling = np.load(path + '-labelling.npy')

weights = np.array([0.1, 0.9])

model = load_model('six_conv_10_epochs.h5', custom_objects={'loss': weighted_categorical_crossentropy(weights)})

result = model.predict(sample.reshape(1, 28, 28, 28, 1))
result = result.reshape(28, 28, 28, 2)
decat_result = np.argmax(result, axis=3)

labelling = decat_result[20, :, :]

masked_data = np.ma.masked_where(labelling == 0, labelling)

plt.imshow(sample[20, :, :], interpolation="none", origin='lower')
plt.imshow(masked_data, interpolation="none", origin='lower', cmap=cm.jet, alpha=0.5)
plt.show()

print(np.bincount(sample_labelling.reshape(-1).astype(int)))
print(np.bincount(decat_result.reshape(-1).astype(int)))

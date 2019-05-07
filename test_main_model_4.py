import numpy as np
from keras.models import load_model
from losses_and_metrics.keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utility_functions import opening_files

volume = opening_files.read_nii("datasets/spine-1/patient0088/2684937/2684937.nii.gz")

weights = np.array([0.1, 0.9])

model = load_model('model_files/main-model.h5', custom_objects={'loss': weighted_categorical_crossentropy(weights)})

output = np.zeros(volume.shape)

for x in range(0, volume.shape[0] - 28, 28):
    for y in range(0, volume.shape[1] - 28, 28):
        for z in range(0, volume.shape[2] - 28, 28):
            corner_a = [x, y, z]
            corner_b = [x + 28, y + 28, z + 28]
            patch = volume[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]
            result = model.predict(patch.reshape(1, 28, 28, 28, 1))
            result = result.reshape(28, 28, 28, 2)
            decat_result = np.argmax(result, axis=3)
            output[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]] = decat_result
            print(x, y, z, np.bincount(decat_result.reshape(-1).astype(int)))


print(volume.shape)
volume_slice = volume[35, :, :]
predict_slice = output[35, :, :]

masked_data = np.ma.masked_where(predict_slice == 0, predict_slice)

plt.imshow(volume_slice.T, interpolation="none", origin='lower')
plt.imshow(masked_data.T, interpolation="none", origin='lower', cmap=cm.jet, alpha=0.5)
plt.show()



import numpy as np
from keras.models import load_model
from losses_and_metrics.keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy
import matplotlib.pyplot as plt
from utility_functions import opening_files

volume = opening_files.read_nii("datasets/spine-1/patient0088/2684937/2684937.nii.gz")

weights = np.array([0.1, 0.9])
model = load_model('model_files/main-model.h5', custom_objects={'loss': weighted_categorical_crossentropy(weights)})

output = np.zeros(volume.shape)

patch_size = np.array([28, 28, 28])

for x in range(0, volume.shape[0] - patch_size[0], patch_size[0]):
    for y in range(0, volume.shape[1] - patch_size[1], patch_size[1]):
        for z in range(0, volume.shape[2] - patch_size[2], patch_size[2]):
            corner_a = [x, y, z]
            corner_b = corner_a + patch_size
            patch = volume[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]
            patch = np.expand_dims(patch, axis=0)
            patch = np.expand_dims(patch, axis=4)
            result = model.predict(patch)
            result = np.squeeze(result, axis=0)
            decat_result = np.argmax(result, axis=3)
            output[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]] = decat_result
            print(x, y, z, np.bincount(decat_result.reshape(-1).astype(int)))


print(volume.shape)
volume_slice = volume[35, :, :]
predict_slice = output[35, :, :]

masked_data = np.ma.masked_where(predict_slice == 0, predict_slice)

plt.imshow(volume_slice.T, interpolation="none", origin='lower')
plt.imshow(masked_data.T, interpolation="none", origin='lower', alpha=0.5)
plt.show()



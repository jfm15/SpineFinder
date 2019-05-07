from packages.UnetCNN.unet3d import training, prediction

from keras.models import load_model
import numpy as np
from utility_functions import opening_files
import matplotlib.pyplot as plt

model = training.load_old_model('main-model.h5')

volume = opening_files.read_nii("datasets/spine-1/patient0078/2551924/2551924.nii.gz")
#volume = np.expand_dims(volume, axis=0)
#volume = np.expand_dims(volume, axis=0)
#result = prediction.patch_wise_prediction(model, volume)

for x in range(0, volume.shape[0] - 28, 28):
    for y in range(0, volume.shape[1] - 28, 28):
        for z in range(0, volume.shape[2] - 28, 28):
            corner_a = [x, y, z]
            corner_b = [x + 28, y + 28, z + 28]
            patch = volume[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]
            result = model.predict(patch.reshape(1, 1, 28, 28, 28))
            result = result.reshape(27, 28, 28, 28)
            decat_result = np.argmax(result, axis=0)
            print(x, y, z, np.unique(decat_result))

'''
patch = volume[:28, :28, :28]
patch = patch.reshape(1, 1, 28, 28, 28)

result = model.predict(patch, verbose=1)
result = result.reshape(27, 28, 28, 28)

decat_result = np.argmax(result, axis=0)

print(np.unique(decat_result))
'''
#np.save("histogram", result)

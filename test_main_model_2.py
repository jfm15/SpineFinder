from keras.models import load_model
import numpy as np
from utility_functions import opening_files
import matplotlib.pyplot as plt

model = load_model('main-model.h5')

volume = opening_files.read_nii("datasets/spine-1/patient0023/4542094/4542094.nii.gz")
patch = volume[:28, :28, :28]
patch = patch.reshape(1, 1, 28, 28, 28)

result = model.predict(patch, verbose=1)

print(result.shape)
np.save("histogram", result)
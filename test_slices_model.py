import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utility_functions import opening_files
from models.six_conv_slices import cool_loss

label_translation = ["B", "C1", "C2", "C3", "C4", "C5", "C6", "C7",
                     "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
                     "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "L6",
                     "S1", "S2"]

# sample = np.load('samples/slices/2684937-2-sample.npy')
# labelling = np.load('samples/slices/2684937-2-labelling.npy')

volume = opening_files.read_nii("datasets/spine-4/patient0286/4557469/4557469.nii.gz")
labels, centroids = opening_files.extract_centroid_info_from_lml("datasets/spine-4/patient0286/4557469/4557469.lml")
centroid_indexes = np.round(centroids / np.array((2.0, 2.0, 2.0))).astype(int)

cut = np.round(np.mean(centroid_indexes[:, 0])).astype(int)

sample = volume[cut, :, :]

model = load_model('model_files/slices_model.h5', custom_objects={'cool_loss': cool_loss})

#i_padding = 4 - sample.shape[0] % 4
#j_padding = 4 - sample.shape[1] % 4
#sample = np.pad(sample, ((0, i_padding), (0, j_padding)), "edge")


'''
for label, centroid_idx in zip(labels, centroid_indexes):
    i, j, k = centroid_idx
    sample_slice = volume[i, :, :]
    prediction = model.predict(sample_slice.reshape(1, *sample_slice.shape, 1))
    prediction = prediction.reshape(*sample_slice.shape)
    print(label, label_translation.index(label), prediction[j, k])
'''

prediction = model.predict(sample.reshape(1, *sample.shape, 1))
prediction = prediction.reshape(*sample.shape)


print(np.unique(prediction))

fig, ax = plt.subplots()

plt.imshow(sample.T)

# plt.imshow(labelling.T, cmap=cm.jet, alpha=0.8)
plt.imshow(prediction.T, cmap=cm.jet, alpha=0.3)
plt.show()

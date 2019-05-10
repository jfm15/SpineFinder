import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sample = np.load('samples/slices/2684937-5-sample.npy')
labelling = np.load('samples/slices/2684937-5-labelling.npy')

model = load_model('model_files/slices_model.h5')

#i_padding = 4 - sample.shape[0] % 4
#j_padding = 4 - sample.shape[1] % 4
#sample = np.pad(sample, ((0, i_padding), (0, j_padding)), "edge")

prediction = model.predict(sample.reshape(1, *sample.shape, 1))

prediction = prediction.reshape(*sample.shape)

print(np.unique(prediction))

fig, ax = plt.subplots()

ax.imshow(sample.T)

# plt.imshow(labelling.T, cmap=cm.jet, alpha=0.2)
# plt.imshow(prediction.T, cmap=cm.jet, alpha=0.5)
plt.show()
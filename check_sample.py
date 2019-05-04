import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sample_path = "samples/2665969-3-"

sample = np.load(sample_path + "sample.npy")
dense_labelling = np.load(sample_path + "labelling.npy")

slice = sample[10, :, :]
labelling = dense_labelling[10, :, :]

masked_data = np.ma.masked_where(labelling == 0, labelling)

fig, ax = plt.subplots(1)
ax.imshow(slice.T, interpolation="none", aspect=8, origin='lower')
plt.imshow(masked_data.T, interpolation="none", aspect=8, origin='lower', cmap=cm.jet, alpha=0.5)
plt.show()
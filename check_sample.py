import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sample_path = "samples/3164509-74-"

sample = np.load(sample_path + "sample.npy")
dense_labelling = np.load(sample_path + "labelling.npy")

slice = sample[15, :, :]
labelling = dense_labelling[15, :, :]

print(np.unique(dense_labelling))
print(np.bincount(dense_labelling.reshape(-1).astype(int)))

masked_data = np.ma.masked_where(labelling == 0, labelling)

fig, ax = plt.subplots(1)
ax.imshow(slice.T, interpolation="none", origin='lower')
plt.imshow(masked_data.T, interpolation="none", origin='lower', cmap=cm.jet, alpha=1)
plt.show()
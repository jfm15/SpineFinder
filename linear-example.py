import numpy as np
import imcut.pycut as pspc
import matplotlib.pyplot as plt
from utility_functions import opening_files
from utility_functions import sampling_helper_functions

spacing = (1.0, 1.0, 1.0)
path = "datasets/spine-2/patient0101/3109354/3109354"
volume = opening_files.read_nii(path + ".nii.gz", spacing=spacing)
labels, centroids = opening_files.extract_centroid_info_from_lml(path + ".lml")
centroids = centroids / np.array(spacing)
idx = 8
chosen_centroid = np.round(centroids[idx]).astype(int)

volume_slice = volume[chosen_centroid[0], :, :]

labelling = sampling_helper_functions.densely_label(volume.shape, labels, centroids, False)

# Show results
colormap = plt.cm.get_cmap('brg')
colormap._init()
colormap._lut[:1:,3]=0


plt.imshow(volume_slice.T, cmap='gray', origin="lower")
plt.imshow(labelling[chosen_centroid[0], :, :].T, cmap=colormap, interpolation='none', alpha=0.4, origin="lower")
plt.show()
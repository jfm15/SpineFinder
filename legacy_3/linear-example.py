import numpy as np
import elasticdeform
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utility_functions import opening_files
from utility_functions import sampling_helper_functions

spacing = (1.0, 1.0, 1.0)
path = "datasets/spine-1/patient0001/2805012/2805012"
volume = opening_files.read_nii(path + ".nii.gz", spacing=spacing)
labels, centroids = opening_files.extract_centroid_info_from_lml(path + ".lml")
centroids = centroids / np.array(spacing)
idx = 7
chosen_centroid = np.round(centroids[idx]).astype(int)

labelling = sampling_helper_functions.densely_label(volume.shape, labels,
                                                    centroids, spacing=(1.0, 1.0, 1.0), use_labels=True)

volume_slice = volume[chosen_centroid[0], :, :]
labelling_slice = labelling[chosen_centroid[0], :, :]

[volume_slice_deformed, labelling_slice_deformed] = elasticdeform.deform_random_grid([volume_slice, labelling_slice],
                                                                                     sigma=5, points=3, order=0)

masked_data = np.ma.masked_where(labelling_slice_deformed == 0, labelling_slice_deformed)

plt.imshow(volume_slice_deformed.T, cmap='gray', origin="lower")
plt.imshow(masked_data.T, cmap=cm.jet, vmin=1, vmax=26, interpolation='none', alpha=0.4, origin="lower")
plt.show()
import numpy as np
import elasticdeform
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

labelling = sampling_helper_functions.densely_label(volume.shape, labels,
                                                    centroids, spacing=(1.0, 1.0, 1.0), use_labels=False)

volume_slice = volume[chosen_centroid[0], :, :]
labelling_slice = labelling[chosen_centroid[0], :, :]

print(volume_slice.shape, labelling_slice.shape)

[volume_slice_deformed, labelling_slice_deformed] = elasticdeform.deform_random_grid([volume_slice, labelling_slice], sigma=5, points=3)
[volume_slice_deformed2, labelling_slice_deformed2] = elasticdeform.deform_random_grid([volume_slice, labelling_slice], sigma=5, points=3)
[volume_slice_deformed3, labelling_slice_deformed3] = elasticdeform.deform_random_grid([volume_slice, labelling_slice], sigma=5, points=3)

print(volume_slice_deformed.shape, labelling_slice_deformed.shape)

# Show results
colormap = plt.cm.get_cmap('brg')
colormap._init()
colormap._lut[:1:,3]=0


plt.imshow(volume_slice_deformed.T, cmap='gray', origin="lower")
plt.imshow(labelling_slice_deformed.T, cmap=colormap, interpolation='none', alpha=0.4, origin="lower")
plt.show()

plt.imshow(volume_slice_deformed2.T, cmap='gray', origin="lower")
plt.imshow(labelling_slice_deformed2.T, cmap=colormap, interpolation='none', alpha=0.4, origin="lower")
plt.show()

plt.imshow(volume_slice_deformed3.T, cmap='gray', origin="lower")
plt.imshow(labelling_slice_deformed3.T, cmap=colormap, interpolation='none', alpha=0.4, origin="lower")
plt.show()
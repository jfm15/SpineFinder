import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import helper_functions as hf

patient1_1_fn = './spine-1/patient0013/4511471/4511471.nii'

centroids_file = open('./spine-1/patient0013/4511471/4511471.lml', 'r')
iter_centroids_file = iter(centroids_file)
next(iter_centroids_file)

centroids = []
for centroid_line in iter_centroids_file:
    centroid_line_split = centroid_line.split()
    centroid_tuple = (centroid_line_split[1].split("_")[0], centroid_line_split[2:5])
    centroids.append(centroid_tuple)

# centroids is in the format [name, [transverse, sagittal, vertical], ...]


sitk_patient1_1 = sitk.ReadImage(patient1_1_fn)
t1 = sitk.GetArrayFromImage(sitk_patient1_1)

# t1 comes in here in the format [vertical, sagittal, transverse] so we transpose
t1 = t1.T
# t1 is now in the shape [transverse, sagittal, vertical]

# transverse axis, sagittal axis, vertical axis
scales = np.array([0.3125, 0.3125, 2.5])
# C1...C7, T1...T12, L1...L5, S1, S2
vertebrae_radii = np.array([])

real_centroids = np.array([[float(i) for i in centroid[1]] for centroid in centroids])
scaled_centroids = real_centroids / scales

radii = 13

info = np.zeros(t1.shape)

for idx, real_centroid in enumerate(real_centroids):
    # we want to check every pixel inside the square where the lengths of the sides are 2 * radii
    real_lower_bounds = real_centroid - radii
    real_upper_bounds = real_centroid + radii

    idx_lower_bounds = hf.real_to_indexes(real_lower_bounds, scales)
    idx_upper_bounds = hf.real_to_indexes(real_upper_bounds, scales)

    idx_lower_bounds = np.maximum(idx_lower_bounds, 0)
    idx_upper_bounds = np.minimum(idx_upper_bounds, t1.shape)

    for t in range(idx_lower_bounds[0], idx_upper_bounds[0]):
        for s in range(idx_lower_bounds[1], idx_upper_bounds[1]):
            for v in range(idx_lower_bounds[2], idx_upper_bounds[2]):
                image_point = [t, s, v]
                real_point = hf.indexes_to_real(image_point, scales)
                if np.linalg.norm(real_point-real_centroid) < radii:
                    info[t, s, v] = idx + 1

    print(idx)

best_transverse_cut = int(round(np.mean(scaled_centroids[:, 0])))
t1_slice = t1[best_transverse_cut, :, :]
t1_slice_info = info[best_transverse_cut, :, :]

masked_data = np.ma.masked_where(t1_slice_info==0, t1_slice_info)

# imshow takes a matrix n x m and plots n up the y axis so we transpose it
plt.imshow(t1_slice.T, interpolation="none", aspect=8, origin='lower')
plt.imshow(masked_data.T, interpolation="none", aspect=8, origin='lower', cmap=cm.jet, alpha=1)

plt.show()
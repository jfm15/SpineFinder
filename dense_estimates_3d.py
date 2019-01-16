import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

patient1_1_fn = './spine-1/patient0013/4511471/4511471.nii'

centroids_file = open('./spine-1/patient0013/4511471/4511471.lml', 'r')
iter_centroids_file = iter(centroids_file)
next(iter_centroids_file)

centroids = []
for centroid_line in iter_centroids_file:
    centroid_line_split = centroid_line.split()
    centroid_tuple = (centroid_line_split[1].split("_")[0], centroid_line_split[2:5])
    centroids.append(centroid_tuple)

sitk_patient1_1 = sitk.ReadImage(patient1_1_fn)

t1 = sitk.GetArrayFromImage(sitk_patient1_1)

print(t1.shape)

# transverse axis, sagittal axis, vertical axis
scales = np.array([0.3125, 0.3125, 2.5])
# C1...C7, T1...T12, L1...L5, S1, S2
vertebrae_radii = np.array([])

real_centroids = np.array([[float(i) for i in centroid[1]] for centroid in centroids])
scaled_centroids = np.array([[float(i) for i in centroid[1]]/scales for centroid in centroids])

radii = 13

info = np.zeros(t1.shape)

for idx, real_centroid in enumerate(real_centroids):
    # we want to check every pixel inside the square where the lengths of the sides are 2 * radii
    real_lower_bound_transverse = real_centroid[0] - radii
    real_upper_bound_transverse = real_centroid[0] + radii
    real_lower_bound_saggital = real_centroid[1] - radii
    real_upper_bound_saggital = real_centroid[1] + radii
    real_lower_bound_vertical = real_centroid[2] - radii
    real_upper_bound_vertical = real_centroid[2] + radii

    idx_lower_bound_transverse = int(max(round(real_lower_bound_transverse / scales[0]), 0))
    idx_upper_bound_transverse = int(min(round(real_upper_bound_transverse / scales[0]), t1.shape[2]))
    idx_lower_bound_saggital = int(max(round(real_lower_bound_saggital / scales[1]), 0))
    idx_upper_bound_saggital = int(min(round(real_upper_bound_saggital / scales[1]), t1.shape[1]))
    idx_lower_bound_vertical = int(max(round(real_lower_bound_vertical / scales[2]), 0))
    idx_upper_bound_vertical = int(min(round(real_upper_bound_vertical / scales[2]), t1.shape[0]))

    print("centroid", real_centroid)
    print(idx_lower_bound_transverse, idx_upper_bound_transverse, idx_lower_bound_saggital, idx_upper_bound_saggital, idx_lower_bound_vertical, idx_upper_bound_vertical)

    for t in range(idx_lower_bound_transverse, idx_upper_bound_transverse):
        for s in range(idx_lower_bound_saggital, idx_upper_bound_saggital):
            for v in range(idx_lower_bound_vertical, idx_upper_bound_vertical):
                image_point = [t, s, v]
                real_point = image_point * scales
                if np.linalg.norm(real_point-real_centroid) < radii:
                    info[v, s, t] = idx + 1

    print(idx)

#best_sagittal_cut = np.mean(scaled_centroids[:, 0])
#best_sagittal_cut = scaled_centroids[10, 0]
t1_slice = t1[int(round(scaled_centroids[16, 2])), :, :]
t1_slice_info = info[int(round(scaled_centroids[16, 2])), :, :]

masked_data = np.ma.masked_where(t1_slice_info==0, t1_slice_info)

plt.imshow(t1_slice, interpolation="none", aspect=1, origin='lower')
plt.imshow(masked_data, interpolation="none", aspect=1, origin='lower', cmap=cm.jet, alpha=1)

plt.show()
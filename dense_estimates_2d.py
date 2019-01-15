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

scales = np.array([0.3125, 0.3125, 2.5])
# C1...C7, T1...T12, L1...L5, S1, S2
vertebrae_radii = np.array([])

real_centroids = np.array([[float(i) for i in centroid[1]] for centroid in centroids])
scaled_centroids = np.array([[float(i) for i in centroid[1]]/scales for centroid in centroids])

best_sagittal_cut = np.mean(scaled_centroids[:, 0])
#best_sagittal_cut = scaled_centroids[2, 0]

t1_sagittal = t1[:, :, int(round(best_sagittal_cut))]

#iterate through every point in t1 sagittal and get likelihood
info = np.zeros(t1_sagittal.shape[0:2])

radii = 13

print(info.shape)

for idx, real_centroid in enumerate(real_centroids):
    # we want to check every pixel inside the square where the lengths of the sides are 2 * radii
    real_lower_bound_x = real_centroid[1] - radii
    real_upper_bound_x = real_centroid[1] + radii
    real_lower_bound_y = real_centroid[2] - radii
    real_upper_bound_y = real_centroid[2] + radii

    idx_lower_bound_x = int(max(round(real_lower_bound_x / scales[1]), 0))
    idx_upper_bound_x = int(min(round(real_upper_bound_x / scales[1]), t1_sagittal.shape[1]))
    idx_lower_bound_y = int(max(round(real_lower_bound_y / scales[2]), 0))
    idx_upper_bound_y = int(min(round(real_upper_bound_y / scales[2]), t1_sagittal.shape[0]))

    for x in range(idx_lower_bound_x, idx_upper_bound_x):
        for y in range(idx_lower_bound_y, idx_upper_bound_y):
            image_point = [int(round(best_sagittal_cut)), x, y]
            real_point = image_point * scales
            if np.linalg.norm(real_point-real_centroid) < radii:
                info[y, x] = idx + 1

    """
    for x in range(-radii, radii):
        for y in range(-radii, radii):


            real_x = real_centroid[0] + x
            real_y = real_centroid[1] + y
            image_point = [int(round(best_sagittal_cut)), real_y, real_x]
            real_point = image_point * scales
            if np.linalg.norm(real_point-real_centroid) < radii:
                print(idx + 1)
                info[real_x, real_y] = idx + 1
    """


"""
for x in range(info.shape[0]):
    for y in range(info.shape[1]):
        for v in range(0, len(centroids)):
            #image_point = [int(round(scaled_centroids[v][0])), y, x]
            image_point = [int(round(best_sagittal_cut)), y, x]
            real_point = image_point * scales
            if np.linalg.norm(real_point-real_centroids[v]) < 13:
                info[x, y] = v + 1
"""

masked_data = np.ma.masked_where(info==0, info)

#print(t1_sagittal.shape)

#print(t1_sagittal)

plt.imshow(t1_sagittal, interpolation="none", aspect=8, origin='lower')
plt.imshow(masked_data, interpolation="none", aspect=8, origin='lower', cmap=cm.jet, alpha=1)

for s_c in scaled_centroids:
    X = s_c[1]
    Y = s_c[2]
    #plt.scatter(X, Y, color="red")

plt.show()
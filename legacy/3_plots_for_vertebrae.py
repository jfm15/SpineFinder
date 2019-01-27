import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

patient1_1_fn = './spine-1/patient0001/2804506/2804506.nii'

centroids_file = open('./spine-1/patient0001/2804506/2804506.lml', 'r')
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

# select vertebrae
idx = 2
label = centroids[idx][0]
scaled_centroid = np.array(list(map(float, centroids[idx][1]))) / scales
print(label, scaled_centroid)

# 2.5mm, 0.3125mm, 0.3125mm
t1_axial = t1[int(round(scaled_centroid[2])), :, :]
t1_coronal = t1[:, int(round(scaled_centroid[1])), :]
t1_sagittal = t1[:, :, int(round(scaled_centroid[0]))]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(t1_axial, interpolation="none", aspect=1, origin='lower')
ax2.imshow(t1_coronal, interpolation="none", aspect=8, origin='lower')
ax3.imshow(t1_sagittal, interpolation="none", aspect=8, origin='lower')


def plot_axis(ax, axis, title):
    X = scaled_centroid[axis[0]]
    Y = scaled_centroid[axis[1]]
    ax.annotate(label, (X, Y), color="red")
    ax.scatter(X, Y, color="red")
    ax.set_title(title)


plot_axis(ax1, (0, 1), "axial")
plot_axis(ax2, (0, 2), "coronal")
plot_axis(ax3, (1, 2), "sagittal")

plt.show()
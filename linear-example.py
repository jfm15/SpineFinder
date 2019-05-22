import numpy as np
import imcut.pycut as pspc
import matplotlib.pyplot as plt
from utility_functions import opening_files

spacing = (1.0, 1.0, 1.0)
path = "datasets/spine-2/patient0101/3109354/3109354"
volume = opening_files.read_nii(path + ".nii.gz", spacing=spacing)
labels, centroids = opening_files.extract_centroid_info_from_lml(path + ".lml")
centroids = centroids / np.array(spacing)
idx = 8
chosen_centroid = np.round(centroids[idx]).astype(int)

volume_slice = volume[:, :, chosen_centroid[2]]

# Make seeds
seeds = np.zeros(volume.shape)

'''
for i in range(centroids.shape[0] - 1):
    dist = np.linalg.norm(centroids[i + 1] - centroids[i])
    spline = np.round(np.linspace(centroids[i], centroids[i + 1], num=np.round(dist).astype(int) * 2)).astype(int)
    for center_point in spline:
        for x in range(-width, width):
            for y in range(-depth, depth):
                point = center_point + np.array([x, y, 0])
                point = np.clip(point, a_min=np.zeros(3), a_max=volume.shape - np.ones(3)).astype(int)
                if np.linalg.norm(np.array([x, y])) <= width:
                    seeds[point[0], point[1], point[2]] = 1
'''

depths = {
    'C1': 14.0,
    'C2': 15.0,
    'C3': 16.5,
    'C4': 17.0,
    'C5': 17.0,
    'C6': 19.0,
    'C7': 20.0,
    'T1': 19.3,
    'T2': 20.0,
    'T3': 22.0,
    'T4': 24.0,
    'T5': 25.0,
    'T6': 27.0,
    'T7': 29.0,
    'T8': 31.0,
    'T9': 33.0,
    'T10': 32.0,
    'T11': 33.5,
    'T12': 34.0,
    'L1': 34.0,
    'L2': 37.0,
    'L3': 38.0,
    'L4': 36.0,
    'L5': 34.0,
    'L6': 34.0,
    'S1': 34.0,
    'S2': 34.0
}


def create_tube(a, b, label):
    dist = np.linalg.norm(b - a)
    spline = np.round(np.linspace(a, b, num=np.round(dist).astype(int) * 2)).astype(int)
    for center_point in spline:
        radius = np.round(depths[label] / 2.0).astype(int)
        for x in range(-radius - 1, radius + 1):
            for y in range(-radius, radius):
                point = center_point + np.array([x, y, 0])
                point = np.clip(point, a_min=np.zeros(3), a_max=volume.shape - np.ones(3)).astype(int)
                if np.linalg.norm(np.array([x, y])) <= radius + 1:
                    seeds[point[0], point[1], point[2]] = 1


# do middle centroids
for i, label in enumerate(labels[1:-1]):
    a = (centroids[i] + centroids[i + 1]) / 2.0
    b = (centroids[i + 1] + centroids[i + 2]) / 2.0
    create_tube(a, b, label)

# do first centroid
b = (centroids[0] + centroids[1]) / 2.0
a = centroids[0] - (b - centroids[0])
a = np.clip(a, a_min=np.zeros(3), a_max=volume.shape - np.ones(3)).astype(int)
create_tube(a, b, labels[0])

# do last centroid
b = (centroids[-2] + centroids[-1]) / 2.0
a = centroids[-1] - (b - centroids[-1])
a = np.clip(a, a_min=np.zeros(3), a_max=volume.shape - np.ones(3)).astype(int)
create_tube(a, b, labels[-1])



# Show results
colormap = plt.cm.get_cmap('brg')
colormap._init()
colormap._lut[:1:,3]=0


plt.imshow(volume_slice.T, cmap='gray', origin="lower")
plt.imshow(seeds[:, :, chosen_centroid[2]].T, cmap=colormap, interpolation='none', alpha=0.4, origin="lower")
plt.show()
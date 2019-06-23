import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import elasticdeform
from utility_functions import opening_files
from utility_functions.labels import LABELS_NO_B, LABELS_NO_B_OR_L6, LABELS_NO_L6
from utility_functions.sampling_helper_functions import densely_label, spherical_densely_label, pre_compute_disks


def vertebrae_counts(dataset_dir, file_ext=".lml"):
    paths = glob.glob(dataset_dir + "/**/*" + file_ext, recursive=True)

    total_count = 0
    frequencies = np.zeros(len(LABELS_NO_B))

    for labelling_path in paths:
        labels, centroids = opening_files.extract_centroid_info_from_lml(labelling_path)
        idx = len(labels) - 1
        total_count += idx
        frequencies[idx] += 1

    print(total_count, frequencies)


def vertebrae_frequencies(dataset_dir, file_ext=".lml"):
    paths = glob.glob(dataset_dir + "/**/*" + file_ext, recursive=True)

    frequencies = np.zeros(len(LABELS_NO_B_OR_L6))

    for labelling_path in paths:
        labels, centroids = opening_files.extract_centroid_info_from_lml(labelling_path)
        for label in labels:
            if label != 'L6':
                idx = LABELS_NO_B_OR_L6.index(label)
                frequencies[idx] += 1

    x = np.arange(len(LABELS_NO_B_OR_L6))
    plt.figure(figsize=(20, 10))
    plt.bar(x, frequencies, 0.7)
    plt.xticks(x, LABELS_NO_B_OR_L6)
    plt.savefig('plots/vertebrae_frequencies_in_dataset.png')


def vertebrae_frequencies_in_samples(samples_dir, plot_path, file_ext="-labelling.npy"):
    paths = glob.glob(samples_dir + "/**/*" + file_ext, recursive=True)

    frequencies = np.zeros(len(LABELS_NO_B_OR_L6))

    for labelling_path in paths:
        labelling = np.load(labelling_path)
        unique_labels_idx = np.round(np.unique(labelling)).astype(int)
        for label_idx in unique_labels_idx:
            if label_idx > 0:
                frequencies[label_idx - 1] += 1

    x = np.arange(len(LABELS_NO_B_OR_L6))
    plt.bar(x, frequencies, 0.5)
    plt.xticks(x, LABELS_NO_B_OR_L6)
    plt.savefig(plot_path + '/vertebrae_frequencies_in_samples.png')


def vertebrae_pixel_frequencies_in_samples(samples_dir, plot_path, file_ext="-labelling.npy"):
    paths = glob.glob(samples_dir + "/**/*" + file_ext, recursive=True)

    frequencies = np.zeros(len(LABELS_NO_B_OR_L6))

    for labelling_path in paths:
        labelling = np.load(labelling_path)
        bincounts = np.bincount(labelling.reshape(-1).astype(int), minlength=len(LABELS_NO_L6))
        frequencies += bincounts[1:]

    x = np.arange(len(LABELS_NO_B_OR_L6))
    plt.bar(x, frequencies, 0.5)
    plt.xticks(x, LABELS_NO_B_OR_L6)
    plt.savefig(plot_path + '/vertebrae_pixel_frequencies_in_samples.png')
    

def old_dense_label_method(scan_path, ext=".nii.gz"):
    scan_path_without_ext = scan_path[:-len(ext)]
    metadata_path = scan_path_without_ext + ".lml"
    
    volume = opening_files.read_nii(scan_path, spacing=(1.0, 1.0, 1.0))
    labels, centroids = opening_files.extract_centroid_info_from_lml(metadata_path)
    dense_labelling = spherical_densely_label(volume.shape, 13, labels, centroids, True)

    # disk_indices = pre_compute_disks((1.0, 1.0, 1.0))
    # dense_labelling = densely_label(volume.shape, disk_indices, labels, centroids, True)

    cut = np.round(np.mean(np.array(centroids)[:, 0])).astype(int)
    # cut = np.round(np.array(centroids)[10, 2]).astype(int)

    volume_slice = volume[cut, :, :]
    labelling_slice = dense_labelling[cut, :, :]

    masked_data = np.ma.masked_where(labelling_slice == 0, labelling_slice)

    fig, ax = plt.subplots(1)

    ax.imshow(volume_slice.T, interpolation="none", origin='lower', cmap='gray', vmin=-2)
    # ax.imshow(masked_data.T, interpolation="none", origin='lower', cmap=cm.jet, vmin=1, vmax=26, alpha=0.4)
    plt.show()


def old_dense_label_method_with_boxes(scan_path, box_coords, ext=".nii.gz"):
    scan_path_without_ext = scan_path[:-len(ext)]
    metadata_path = scan_path_without_ext + ".lml"

    volume = opening_files.read_nii(scan_path, spacing=(1.0, 1.0, 1.0))
    labels, centroids = opening_files.extract_centroid_info_from_lml(metadata_path)
    dense_labelling = spherical_densely_label(volume.shape, 13, labels, centroids, True)

    cut = np.round(np.mean(np.array(centroids)[:, 0])).astype(int)

    volume_slice = volume[cut]
    labelling_slice = dense_labelling[cut]

    masked_data = np.ma.masked_where(labelling_slice == 0, labelling_slice)

    fig, ax = plt.subplots(1)

    ax.imshow(volume_slice.T, interpolation="none", origin='lower', cmap='gray')
    ax.imshow(masked_data.T, interpolation="none", origin='lower', cmap=cm.jet, vmin=1, vmax=26, alpha=0.4)

    for coords in box_coords:
        rect = patches.Rectangle(coords[:2], coords[2], coords[3], linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)

    plt.show()


def old_dense_label_method_patch(scan_path, coords, ext=".nii.gz"):
    scan_path_without_ext = scan_path[:-len(ext)]
    metadata_path = scan_path_without_ext + ".lml"

    volume = opening_files.read_nii(scan_path, spacing=(1.0, 1.0, 1.0))
    labels, centroids = opening_files.extract_centroid_info_from_lml(metadata_path)

    # disk_indices = pre_compute_disks((1.0, 1.0, 1.0))
    # dense_labelling = densely_label(volume.shape, disk_indices, labels, centroids, True)
    dense_labelling = spherical_densely_label(volume.shape, 13, labels, centroids, True)

    cut = np.round(np.mean(np.array(centroids)[:, 0])).astype(int)
    # cut = np.round(np.array(centroids)[1, 2]).astype(int)

    volume_slice = volume[cut]
    labelling_slice = dense_labelling[cut]

    '''
    [volume_slice, labelling_slice] = elasticdeform.deform_random_grid(
        [volume_slice, labelling_slice], sigma=7, points=3, order=0)
    '''


    volume_slice = volume_slice[coords[0]:coords[0]+coords[2], coords[1]:coords[1]+coords[3]]
    labelling_slice = labelling_slice[coords[0]:coords[0]+coords[2], coords[1]:coords[1]+coords[3]]

    masked_data = np.ma.masked_where(labelling_slice == 0, labelling_slice)

    fig, ax = plt.subplots(1)

    ax.imshow(volume_slice.T, interpolation="none", origin='lower', cmap='gray')
    # ax.imshow(masked_data.T, interpolation="none", origin='lower', cmap=cm.jet, vmin=1, vmax=26, alpha=0.4)

    '''
    for label, centroid in zip(labels, centroids):
        X = centroid[0]
        Y = centroid[2]
        if X > coords[0] and X < coords[0]+coords[2]:
            if Y > coords[1] and Y < coords[1]+coords[3]:
                plt.annotate(label, (X, Y), color="white", fontsize=6)
                plt.scatter(X, Y, color="white", s=4)
    '''

    plt.show()


def plot_relu():
    xs = np.linspace(-5, 5, 11)
    ys = np.maximum(0, xs)
    print(xs, ys)
    plt.title("ReLu function")
    plt.plot(xs, ys)
    plt.show()


def show_labels(scan_path, ext=".nii.gz"):
    scan_path_without_ext = scan_path[:-len(ext)]
    metadata_path = scan_path_without_ext + ".lml"

    volume = opening_files.read_nii(scan_path, spacing=(1.0, 1.0, 1.0))
    labels, centroids = opening_files.extract_centroid_info_from_lml(metadata_path)
    centroids = np.array(centroids)

    cut = np.round(np.mean(centroids[:, 0])).astype(int)

    volume_slice = volume[cut]

    plt.imshow(volume_slice.T, interpolation="none", origin='lower', cmap='gray', vmin=-2)

    '''
    b = (centroids[0] + centroids[1]) / 2.0
    a = 2 * centroids[0] - b
    middles = [a]

    # do middle centroids
    for i, label in enumerate(labels[:-1]):
        middles.append((centroids[i] + centroids[i + 1]) / 2.0)

    b = (centroids[-2] + centroids[-1]) / 2.0
    a = centroids[-1] - (b - centroids[-1])
    a = np.clip(a, a_min=np.zeros(3), a_max=volume.shape - np.ones(3)).astype(int)
    middles.append(a)
    middles = np.array(middles)
    xs = middles[:, 1]
    ys = middles[:, 2]
    plt.plot(xs, ys, 'r')
    plt.ylim(0, 475)

    for middle in middles:
        X = middle[1]
        Y = middle[2]
        # plt.annotate(label, (X, Y), color="red", fontsize=6)
        plt.scatter(X, Y, color="red", s=16)

    '''

    for label, centroid in zip(labels, centroids):
        X = centroid[1]
        Y = centroid[2]
        plt.annotate(label, (X, Y), color="red", fontsize=6)
        plt.scatter(X, Y, color="red", s=8)

    plt.plot(centroids[:, 1], centroids[:, 2], linewidth=1.0, color="red")

    plt.show()


def test_box_plot():
    data1 = np.random.rand(50) * 100
    data2 = np.random.rand(50) * 100
    plt.boxplot([data1, data2], labels=["hello", "world"])
    plt.show()


#vertebrae_counts('datasets')
#vertebrae_frequencies('datasets')
# vertebrae_frequencies_in_samples('samples/slices', 'plots')
# vertebrae_pixel_frequencies_in_samples('samples/slices', 'plots')
old_dense_label_method("datasets/spine-2/patient0099/2902226/2902226.nii.gz")


# test_box_plot()

'''
old_dense_label_method_with_boxes("datasets/spine-1/patient0088/2684937/2684937.nii.gz", 
                                  np.array([[10, 150, 140, 140]]))
'''

'''
old_dense_label_method_patch("datasets/spine-4/patient0295/3058244/3058244.nii.gz",
                                  np.array([20, 30, 80, 320]))
'''


#plot_relu()

# show_labels("datasets/spine-1/patient0088/2684937/2684937.nii.gz")
# old_dense_label_method("datasets/spine-1/patient0088/2684937/2684937.nii.gz")

# 5 vertebrae in datasets/spine-1/patient0003/4614554/4614554.nii.gz
# this patient has some implants datasets/spine-2/patient0091/4543202/4543202.nii.gz
# this patient has a twisted spine datasets/spine-2/patient0101/3109354/3109354.nii.gz
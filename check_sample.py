import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm


def check_multi_class_sample(sample_path):
    sample_path_without_ext = sample_path[:-len("sample.npy")]
    labelling_path = sample_path_without_ext + "labelling.npy"

    sample = np.load(sample_path)
    labelling = np.load(labelling_path)

    print(np.bincount(labelling.reshape(-1).astype(int)))

    # cut = int(np.round(sample.shape[0] / 2.0))
    cut = 30

    sample_slice = sample[cut, :, :]

    padded_labelling = np.zeros(sample.shape)

    offset = ((np.array(sample.shape) - np.array(labelling.shape)) / 2.0).astype(int)

    corner_a = offset
    corner_b = corner_a + labelling.shape
    rect = patches.Rectangle(corner_a[1:3], labelling.shape[1], labelling.shape[2], linewidth=1, edgecolor='r', facecolor='none')
    padded_labelling[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]] = labelling

    labelling_slice = padded_labelling[cut, :, :]

    masked_data = np.ma.masked_where(labelling_slice == 0, labelling_slice)

    fig, ax = plt.subplots(1)

    ax.imshow(sample_slice.T, interpolation="none", origin='lower', cmap='gray')
    ax.imshow(masked_data.T, interpolation="none", origin='lower', cmap=cm.jet, alpha=0.4)
    ax.add_patch(rect)
    plt.show()


def check_slice_sample(sample_path):
    sample_path_without_ext = sample_path[:-len("sample.npy")]
    labelling_path = sample_path_without_ext + "labelling.npy"

    sample = np.load(sample_path)
    labelling = np.load(labelling_path)

    masked_data = np.ma.masked_where(labelling == 0, labelling)

    plt.imshow(sample.T, interpolation="none", origin='lower', cmap='gray', vmin=-2)
    plt.imshow(masked_data.T, interpolation="none", origin='lower', cmap=cm.jet, vmin=1, vmax=26,  alpha=0.4)
    plt.show()


check_multi_class_sample("samples/two_class/2805012-2-sample.npy")
# check_slice_sample("samples/slices/4533808-5-sample.npy")
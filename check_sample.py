import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def check_multi_class_sample(sample_path):
    sample_path_without_ext = sample_path[:-len("sample.npy")]
    labelling_path = sample_path_without_ext + "labelling.npy"

    sample = np.load(sample_path)
    labelling = np.load(labelling_path)

    print(np.bincount(labelling.reshape(-1).astype(int)))

    # cut = int(np.round(sample.shape[0] / 2.0))
    cut = 1

    sample_slice = sample[cut, :, :]
    labelling_slice = labelling[cut, :, :]

    masked_data = np.ma.masked_where(labelling_slice == 0, labelling_slice)

    plt.imshow(sample_slice.T, interpolation="none", origin='lower')
    plt.imshow(masked_data.T, interpolation="none", origin='lower', cmap=cm.jet, alpha=0.4)
    plt.show()


def check_slice_sample(sample_path):
    sample_path_without_ext = sample_path[:-len("sample.npy")]
    labelling_path = sample_path_without_ext + "labelling.npy"

    sample = np.load(sample_path)
    labelling = np.load(labelling_path)

    masked_data = np.ma.masked_where(labelling == 0, labelling)

    plt.imshow(sample.T, interpolation="none", origin='lower')
    plt.imshow(masked_data.T, interpolation="none", origin='lower', cmap=cm.jet, alpha=0.4)
    plt.show()


# check_multi_class_sample("samples/two_class/2684937-3-sample.npy")
check_slice_sample("samples/slices/2684937-1-sample.npy")
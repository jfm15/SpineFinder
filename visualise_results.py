import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utility_functions import opening_files


def visualise_results(data_path, prediction_path, plot_dir, cut_range=(0.2, 0.8), rows=1, cols=6, file_ext="-prediction.npy"):

    volume = opening_files.read_nii(data_path)
    prediction = np.load(prediction_path)

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(9.3, 6))

    fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.05)

    lower = volume.shape[0] * cut_range[0]
    upper = volume.shape[0] * cut_range[1]
    cut_indices = np.round(np.linspace(lower, upper, num=rows*cols)).astype(int)

    for ax, cut_idx in zip(axs.flat, cut_indices):
        volume_slice = volume[cut_idx, :, :]
        prediction_slice = prediction[cut_idx, :, :]

        ax.imshow(volume_slice.T, vmin=-1000, vmax=1000)

        if np.unique(prediction_slice).shape[0] > 1:
            masked_data = np.ma.masked_where(prediction_slice == 0, prediction_slice)
            ax.imshow(masked_data.T, cmap=cm.spring, alpha=0.5)

    plt.tight_layout()

    ext_len = len(file_ext)
    name = (prediction_path.rsplit('/', 1)[-1])[:-ext_len]
    name = name + "-plots"
    plot_path = '/'.join([plot_dir, name])
    plt.savefig(plot_path)


visualise_results(data_path="datasets/spine-1/patient0088/2684937/2684937.nii.gz",
                  prediction_path="predictions/six_conv_20_epochs/2684937-prediction.npy",
                  plot_dir="plots")
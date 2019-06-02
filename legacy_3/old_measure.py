import numpy as np
import os, glob
import keras_metrics as km
from utility_functions import opening_files, sampling_helper_functions
from keras.models import load_model
from losses_and_metrics.keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy
from models.simple_identification import ignore_background_loss, vertebrae_classification_rate
from losses_and_metrics.dsc import dice_coef_label
from utility_functions.labels import LABELS_NO_L6
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import MeanShift


def apply_ideal_detection(volume, centroid_indexes):

    output = np.zeros(volume.shape)

    for centroid_idx in centroid_indexes:
        for i in range(-10, 10):
            for j in range(-10, 10):
                for k in range(-10, 10):
                    point = np.array(centroid_idx) + np.array([i, j, k])
                    point = np.clip(point, a_min=np.zeros(3), a_max=volume.shape - np.ones(3))
                    dist = np.linalg.norm(point - centroid_idx)
                    if dist < 10:
                        point = point.astype(int)
                        output[point[0], point[1], point[2]] = 1
    return output


def test_individual_scan(scan_path, centroid_path, print_centroids=True, save_centroids=False, centroids_path="",
                         plot_detections=False, detections_path="",
                         save_identifications=False, identifications_path="",
                         save_plots=False, plots_path="", ideal_detection=False,
                         spacing=(2.0, 2.0, 2.0)):
    sub_path = scan_path.split('/', 1)[1]
    sub_path = sub_path[:-len(".nii.gz")]
    sub_path_split = sub_path.split('/')
    dir_path = '/'.join(sub_path_split[:-1])
    name = sub_path_split[-1]

    # print identification_map
    weights = np.array([0.1, 0.9])
    detection_model_objects = {'loss': weighted_categorical_crossentropy(weights),
                             'binary_recall': km.binary_recall()}
    identification_model_objects = {'ignore_background_loss': ignore_background_loss}
    pred_labels, pred_centroid_estimates, pred_detections, pred_identifications = test_scan(
        scan_path=scan_path,
        centroid_path=centroid_path,
        detection_model_path="model_files/two_class_model.h5",
        detection_model_input_shape=np.array([30, 30, 36]),
        detection_model_objects=detection_model_objects,
        identification_model_path="model_files/slices_model.h5",
        identification_model_objects=identification_model_objects,
        ideal_detection=ideal_detection)


    # options
    if print_centroids:
        for label, centroid in zip(pred_labels, pred_centroid_estimates):
            print(label, centroid)

    if save_centroids:
        file_dir_path = '/'.join([centroids_path, dir_path])
        if not os.path.exists(file_dir_path):
            os.makedirs(file_dir_path)
        file_path = file_dir_path + "/" + name + "-pred-centroids"
        file = open(file_path + ".txt", "w")
        for label, centroid in zip(pred_labels, pred_centroid_estimates):
            file.write(" ".join([label, str(centroid[0]), str(centroid[1]), str(centroid[2]), "\n"]))
        file.close()

    pred_centroid_estimates = np.array(pred_centroid_estimates)
    pred_centroid_estimates = pred_centroid_estimates / np.array(spacing)

    if plot_detections:
        detections_dir_path = '/'.join([detections_path, dir_path])
        if not os.path.exists(detections_dir_path):
            os.makedirs(detections_dir_path)

        detection_plot = detections_dir_path + "/" + name + "-detection-plot.png"

        volume = opening_files.read_nii(scan_path)

        # get cuts
        cut = np.mean(pred_centroid_estimates[:, 0])
        cut = np.round(cut).astype(int)

        volume_slice = volume[cut, :, :]
        detections_slice = pred_detections[cut, :, :]

        masked_data = np.ma.masked_where(detections_slice == 0, detections_slice)

        plt.imshow(volume_slice.T)
        plt.imshow(masked_data.T, cmap=cm.jet, alpha=0.3)
        plt.savefig(detection_plot)
        plt.close()

    if save_identifications:
        identifications_dir_path = '/'.join([identifications_path, dir_path])
        if not os.path.exists(identifications_dir_path):
            os.makedirs(identifications_dir_path)
        file_path = identifications_dir_path + "/" + name + "-identifications"
        np.save(file_path, pred_identifications)

    if save_plots:
        plots_dir_path = '/'.join([plots_path, dir_path])
        if not os.path.exists(plots_dir_path):
            os.makedirs(plots_dir_path)
        identification_plot = plots_dir_path + "/" + name + "-id-plot.png"
        centroid_plot = plots_dir_path + "/" + name + "-centroid-plot.png"

        # make plots
        volume = opening_files.read_nii(scan_path)

        # get cuts
        cut = np.mean(pred_centroid_estimates[:, 0])
        cut = np.round(cut).astype(int)

        volume_slice = volume[cut, :, :]
        identifications_slice = pred_identifications[cut, :, :]

        # first plot
        fig1, ax1 = plt.subplots()
        ax1.imshow(volume_slice.T)
        ax1.imshow(identifications_slice.T, cmap=cm.jet, alpha=0.3)
        fig1.savefig(identification_plot)
        plt.close(fig1)

        # second plot
        fig2, ax2 = plt.subplots()
        ax2.imshow(volume_slice.T)

        for label, centroid in zip(pred_labels, pred_centroid_estimates):
            ax2.annotate(label, (centroid[1], centroid[2]), color="red")
            ax2.scatter(centroid[1], centroid[2], s=2, color="red")

        fig2.savefig(centroid_plot)
        plt.close(fig2)


def test_multiple_scans(scans_dir, print_centroids=True, save_centroids=True, plot_detections=True,
                        detections_path="results/detections", centroids_path="results/centroids",
                        save_plots=True, plots_path="results/plots"):

    for scan_path in glob.glob(scans_dir + "/**/*.nii.gz", recursive=True):
        scan_path_without_ext = scan_path[:-len(".nii.gz")]
        centroid_path = scan_path_without_ext + ".lml"

        test_individual_scan(scan_path=scan_path, centroid_path=centroid_path,
                             plot_detections=plot_detections, detections_path=detections_path,
                             print_centroids=print_centroids, save_centroids=save_centroids,
                             centroids_path=centroids_path, save_plots=save_plots, plots_path=plots_path,
                             ideal_detection=False)

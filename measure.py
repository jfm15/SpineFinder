# The aim of this script is to provide measurements for any part of the pipeline
import numpy as np
import os, glob
import keras_metrics as km
from utility_functions import opening_files, sampling_helper_functions
from keras.models import load_model
from losses_and_metrics.keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy
from models.six_conv_slices import cool_loss
from utility_functions.labels import LABELS
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def apply_detection_model(volume, model, patch_size):

    output = np.zeros(volume.shape)

    for x in range(0, volume.shape[0] - patch_size[0], patch_size[0]):
        for y in range(0, volume.shape[1] - patch_size[1], patch_size[1]):
            for z in range(0, volume.shape[2] - patch_size[2], patch_size[2]):
                corner_a = [x, y, z]
                corner_b = corner_a + patch_size
                patch = volume[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]
                patch = patch.reshape(1, *patch_size, 1)
                result = model.predict(patch)
                result = np.squeeze(result, axis=0)
                decat_result = np.argmax(result, axis=3)
                output[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]] = decat_result
                # print(x, y, z, np.bincount(decat_result.reshape(-1).astype(int)))

    return output


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


def apply_identification_model(volume, bounds, detections, model):
    i_min, i_max, j_min, j_max, k_min, k_max = bounds
    cropped_volume = volume[i_min:i_max, j_min:j_max, k_min:k_max]
    output = np.zeros(volume.shape)

    for i in range(i_max - i_min):
        volume_slice = cropped_volume[i, :, :]
        volume_slice_input = volume_slice.reshape(1, *volume_slice.shape, 1)
        prediction = model.predict(volume_slice_input)
        prediction = prediction.reshape(*volume_slice.shape)
        output[i + i_min, j_min:j_max, k_min:k_max] = prediction

    return output


def test_scan(scan_path, centroid_path, detection_model_path, detection_model_input_shape, detection_model_objects,
              identification_model_path, identification_model_objects, ideal_detection):

    volume = opening_files.read_nii(scan_path)

    # first stage is to put the volume through the detection model to find where vertebrae are
    if not ideal_detection:
        detection_model = load_model(detection_model_path, custom_objects=detection_model_objects)
        detections = apply_detection_model(volume, detection_model, detection_model_input_shape)
    else:
        _, centroids = opening_files.extract_centroid_info_from_lml(centroid_path)
        centroid_indexes = np.round(centroids / np.array((2.0, 2.0, 2.0))).astype(int)
        detections = apply_ideal_detection(volume, centroid_indexes)

    # get the largest island
    # bounds, detections = sampling_helper_functions.crop_labelling(detections)
    largest_island_np = np.transpose(np.nonzero(detections))
    i_min = np.min(largest_island_np[:, 0])
    i_max = np.max(largest_island_np[:, 0])
    j_min = np.min(largest_island_np[:, 1])
    j_max = np.max(largest_island_np[:, 1])
    k_min = np.min(largest_island_np[:, 2])
    k_max = np.max(largest_island_np[:, 2])
    bounds = (i_min, i_max, j_min, j_max, k_min, k_max)

    # second stage is to pass slices of this to the identification network
    identification_model = load_model(identification_model_path, custom_objects=identification_model_objects)
    identifications = apply_identification_model(volume, bounds, detections, identification_model)

    # crop parts of slices
    identifications *= detections

    # aggregate the predictions
    identifications = np.round(identifications).astype(int)
    histogram = {}
    for i in range(identifications.shape[0]):
        for j in range(identifications.shape[1]):
            for k in range(identifications.shape[2]):
                key = identifications[i, j, k]
                if key != 0:
                    if key in histogram:
                        histogram[key] = histogram[key] + [[i, j, k]]
                    else:
                        histogram[key] = [[i, j, k]]

    # find averages
    labels = []
    centroid_estimates = []
    for key in sorted(histogram.keys()):
        if 0 <= key < len(LABELS):
            arr = np.array(histogram[key])
            if arr.shape[0] > 100:
                centroid_estimate = np.mean(arr, axis=0)
                centroid_estimate *= 2
                centroid_estimate = np.around(centroid_estimate, decimals=2)
                labels.append(LABELS[key])
                centroid_estimates.append(list(centroid_estimate))

    return labels, centroid_estimates, detections, identifications


def test_individual_scan(scan_path, centroid_path, print_centroids=True, save_centroids=False, centroids_path="",
                         plot_detections=False, detections_path="",
                         save_identifications=False, identifications_path="",
                         save_plots=False, plots_path="", ideal_detection=False):
    sub_path = scan_path.split('/', 1)[1]
    sub_path = sub_path[:-len(".nii.gz")]
    sub_path_split = sub_path.split('/')
    dir_path = '/'.join(sub_path_split[:-1])
    name = sub_path_split[-1]

    # print identification_map
    weights = np.array([0.1, 0.9])
    detection_model_objects = {'loss': weighted_categorical_crossentropy(weights),
                             'binary_recall': km.binary_recall()}
    identification_model_objects = {'cool_loss': cool_loss}
    pred_labels, pred_centroid_estimates, pred_detections, pred_identifications = test_scan(
        scan_path=scan_path,
        centroid_path=centroid_path,
        detection_model_path="model_files/two_class_model.h5",
        detection_model_input_shape=np.array([30, 30, 60]),
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
    pred_centroid_estimates = pred_centroid_estimates / 2.0

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


test_multiple_scans("datasets_test")
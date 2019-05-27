# The aim of this script is to provide measurements for any part of the pipeline
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


def apply_detection_model(volume, model, X_size, y_size):

    # E.g if X_size = 30 x 30 x 30 and y_size is 20 x 20 x 20
    # Then cropping is ((5, 5), (5, 5), (5, 5)) pad the whole thing by cropping
    # Then pad an additional amount to make it divisible by Y_size + cropping
    # Then iterate through in y_size + cropping steps
    # Then uncrop at the end

    border = ((X_size - y_size) / 2.0).astype(int)
    border_paddings = np.array(list(zip(border, border))).astype(int)
    volume_padded = np.pad(volume, border_paddings, mode="constant")

    # pad to make it divisible to patch size
    divisible_area = volume_padded.shape - X_size
    paddings = np.mod(y_size - np.mod(divisible_area.shape, y_size), y_size)
    paddings = np.array(list(zip(np.zeros(3), paddings))).astype(int)
    volume_padded = np.pad(volume_padded, paddings, mode="constant")

    output = np.zeros(volume_padded.shape)

    print(X_size, y_size, volume.shape, output.shape)

    for x in range(0, volume_padded.shape[0] - X_size[0] + 1, y_size[0]):
        for y in range(0, volume_padded.shape[1] - X_size[1] + 1, y_size[1]):
            for z in range(0, volume_padded.shape[2] - X_size[2] + 1, y_size[2]):
                print(x, y, z)
                corner_a = [x, y, z]
                corner_b = corner_a + X_size
                corner_c = corner_a + border
                corner_d = corner_c + y_size
                patch = volume_padded[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]
                patch = patch.reshape(1, *X_size, 1)
                result = model.predict(patch)
                result = np.squeeze(result, axis=0)
                decat_result = np.argmax(result, axis=3)
                cropped_decat_result = decat_result[border[0]:-border[0], border[1]:-border[1], border[2]:-border[2]]
                output[corner_c[0]:corner_d[0], corner_c[1]:corner_d[1], corner_c[2]:corner_d[2]] = cropped_decat_result
                # print(x, y, z, np.bincount(decat_result.reshape(-1).astype(int)))

    return output[border[0]:border[0] + volume.shape[0],
           border[1]:border[1] + volume.shape[1],
           border[2]:border[2] + volume.shape[2]]


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


def apply_identification_model(volume, i_min, i_max, model, X_size, y_size):

    border = ((X_size - y_size) / 2.0).astype(int)
    border = np.append(0, border)
    border_paddings = np.array(list(zip(border, border))).astype(int)
    volume_padded = np.pad(volume, border_paddings, mode="constant")

    divisible_area = volume_padded.shape[1:3] - X_size
    paddings = np.mod(y_size - np.mod(divisible_area.shape, y_size), y_size)
    paddings = np.append(0, paddings)
    paddings = np.array(list(zip(np.zeros(3), paddings))).astype(int)
    volume_padded = np.pad(volume_padded, paddings, mode="constant")

    print(volume.shape, volume_padded.shape)
    output = np.zeros(volume_padded.shape)

    for i in range(i_min, i_max):
        cnt = (i - i_min) / (i_max - i_min) * 100
        print(str(cnt))
        volume_slice_padded = volume_padded[i, :, :]
        for x in range(0, volume_slice_padded.shape[0] - X_size[0] + 1, y_size[0]):
            for y in range(0, volume_slice_padded.shape[1] - X_size[1] + 1, y_size[1]):
                corner_a = [x, y]
                corner_b = corner_a + X_size
                corner_c = corner_a + border
                corner_d = corner_c + y_size
                patch = volume_slice_padded[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1]]
                patch = patch.reshape(1, *X_size, 1)
                result = model.predict(patch)
                result = np.squeeze(result, axis=0)
                result = np.squeeze(result, axis=-1)
                result = np.round(result)
                print(np.unique(result))
                cropped_result = result[border[0]:-border[0], border[1]:-border[1]]
                output[i, corner_c[0]:corner_d[0], corner_c[1]:corner_d[1]] = cropped_result

        '''
        patch = volume_slice_padded.reshape(1, *volume_slice_padded.shape, 1)
        result = model.predict(patch)
        result = np.squeeze(result, axis=0)
        result = np.squeeze(result, axis=-1)
        result = np.round(result)
        output[i, :, :] = result
        '''

    output = output[:, border[0]:border[0] + volume.shape[1], border[1]:border[1] + volume.shape[2]]
    print(output.shape)
    return output


def test_scan(scan_path, centroid_path, detection_model_path, detection_X_shape, detection_y_shape, detection_model_objects,
              identification_model_path, identification_X_shape, identification_y_shape, identification_model_objects,
              ideal_detection=False, spacing=(2.0, 2.0, 2.0)):

    volume = opening_files.read_nii(scan_path, spacing)

    # first stage is to put the volume through the detection model to find where vertebrae are
    if not ideal_detection:
        detection_model = load_model(detection_model_path, custom_objects=detection_model_objects)
        detections = apply_detection_model(volume, detection_model, detection_X_shape, detection_y_shape)
    else:
        _, centroids = opening_files.extract_centroid_info_from_lml(centroid_path)
        centroid_indexes = np.round(centroids / np.array(spacing)).astype(int)
        detections = apply_ideal_detection(volume, centroid_indexes)

    # get the largest island
    # _, largest_island_np = sampling_helper_functions.crop_labelling(detections)
    largest_island_np = np.transpose(np.nonzero(detections))
    # largest_island_np = np.transpose(np.nonzero(largest_island_np)).astype(int)
    i_min = np.min(largest_island_np[:, 0])
    i_max = np.max(largest_island_np[:, 0])

    # second stage is to pass slices of this to the identification network
    identification_model = load_model(identification_model_path, custom_objects=identification_model_objects)
    identifications = apply_identification_model(volume, i_min, i_max,
                                                 identification_model,
                                                 identification_X_shape,
                                                 identification_y_shape)

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
        if 0 <= key < len(LABELS_NO_L6):
            arr = np.array(histogram[key])
            print(LABELS_NO_L6[key], arr.shape[0])
            if arr.shape[0] > 3000:
                centroid_estimate = np.median(arr, axis=0)
                centroid_estimate = np.around(centroid_estimate, decimals=2)
                labels.append(LABELS_NO_L6[key])
                centroid_estimates.append(list(centroid_estimate))

    return labels, centroid_estimates, detections, identifications


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


def compete_detection_picture(scans_dir, models_dir, plot_path, spacing=(2.0, 2.0, 2.0)):

    scan_paths = glob.glob(scans_dir + "/**/*.nii.gz", recursive=True)
    model_paths = glob.glob(models_dir + "/*.h5", recursive=True)
    no_of_scan_paths = len(scan_paths)
    no_of_model_paths = len(model_paths)
    print("rows", no_of_model_paths, "cols", no_of_scan_paths)

    weights = np.array([0.1, 0.9])
    model_objects = {'loss': weighted_categorical_crossentropy(weights),
                     'binary_recall': km.binary_recall(),
                     'dice_coef': dice_coef_label(label=1)}

    fig, axes = plt.subplots(nrows=no_of_model_paths, ncols=no_of_scan_paths, figsize=(20, 10), dpi=300)

    i = 1

    for col, scan_path in enumerate(scan_paths):

        scan_path_without_ext = scan_path[:-len(".nii.gz")]
        centroid_path = scan_path_without_ext + ".lml"

        _, centroids = opening_files.extract_centroid_info_from_lml(centroid_path)

        scan_name = (scan_path.rsplit('/', 1)[-1])[:-len(".nii.gz")]
        axes[0, col].set_title(scan_name, fontsize=10, pad=10)

        for row, model_path in enumerate(model_paths):
            print(i)

            size = np.array([30, 30, 36])
            current_spacing = spacing
            if model_path == "saved_current_models/detec-15:59.h5" \
                    or model_path == "saved_current_models/detec-15:59-20e.h5" :
                print("here")
                size = np.array([64, 64, 80])
                current_spacing = (1.0, 1.0, 1.0)

            centroid_indexes = centroids / np.array(current_spacing)
            cut = np.round(np.mean(centroid_indexes[:, 0])).astype(int)

            model_name = (model_path.rsplit('/', 1)[-1])[:-len(".h5")]
            axes[row, 0].set_ylabel(model_name, rotation=0, labelpad=50, fontsize=10)

            volume = opening_files.read_nii(scan_path, spacing=current_spacing)
            detection_model = load_model(model_path, custom_objects=model_objects)

            detections = apply_detection_model(volume, detection_model, size)

            volume_slice = volume[cut, :, :]
            detections_slice = detections[cut, :, :]

            masked_data = np.ma.masked_where(detections_slice == 0, detections_slice)

            axes[row, col].imshow(volume_slice.T, cmap='gray')
            axes[row, col].imshow(masked_data.T, cmap=cm.autumn, alpha=0.4)

            i += 1

    fig.subplots_adjust(wspace=-0.2, hspace=0.4)
    fig.savefig(plot_path + '/detection-complete.png')


def complete_identification_picture(scans_dir, detection_model_path, identification_model_path, plot_path,
                                    spacing=(2.0, 2.0, 2.0)):
    scan_paths = glob.glob(scans_dir + "/**/*.nii.gz", recursive=True)[2:4]
    no_of_scan_paths = len(scan_paths)

    weights = np.array([0.1, 0.9])
    detection_model_objects = {'loss': weighted_categorical_crossentropy(weights),
                     'binary_recall': km.binary_recall(),
                     'dice_coef': dice_coef_label(label=1)}

    identification_model_objects = {'ignore_background_loss': ignore_background_loss,
                                    'vertebrae_classification_rate': vertebrae_classification_rate}

    fig, axes = plt.subplots(nrows=1, ncols=no_of_scan_paths, figsize=(30, 6), dpi=300)

    i = 1

    for col, scan_path in enumerate(scan_paths):
        print(i, scan_path)
        scan_path_without_ext = scan_path[:-len(".nii.gz")]
        centroid_path = scan_path_without_ext + ".lml"

        labels, centroids = opening_files.extract_centroid_info_from_lml(centroid_path)
        centroid_indexes = centroids / np.array(spacing)

        cut = np.round(np.mean(centroid_indexes[:, 0])).astype(int)

        scan_name = (scan_path.rsplit('/', 1)[-1])[:-len(".nii.gz")]
        axes[col].set_title(scan_name, fontsize=10, pad=10)

        detection_model_name = (detection_model_path.rsplit('/', 1)[-1])[:-len(".h5")]
        identification_model_name = (identification_model_path.rsplit('/', 1)[-1])[:-len(".h5")]
        name = detection_model_name + "\n" + identification_model_name
        axes[0].set_ylabel(name, rotation=0, labelpad=50, fontsize=10)

        pred_labels, pred_centroid_estimates, pred_detections, pred_identifications = test_scan(
            scan_path=scan_path,
            centroid_path=centroid_path,
            detection_model_path=detection_model_path,
            detection_X_shape=np.array([64, 64, 80]),
            detection_y_shape=np.array([32, 32, 40]),
            detection_model_objects=detection_model_objects,
            identification_model_path=identification_model_path,
            identification_X_shape=np.array([80, 320]),
            identification_y_shape=np.array([40, 160]),
            identification_model_objects=identification_model_objects,
            spacing=spacing)

        volume = opening_files.read_nii(scan_path, spacing=spacing)

        volume_slice = volume[cut, :, :]
        # identifications_slice = pred_identifications[cut, :, :]
        identifications_slice = np.max(pred_identifications, axis=0)

        masked_data = np.ma.masked_where(identifications_slice == 0, identifications_slice)

        axes[col].imshow(volume_slice.T, cmap='gray', origin='lower')
        axes[col].imshow(masked_data.T, vmin=1, vmax=27, cmap=cm.jet, alpha=0.4, origin='lower')

        for label, centroid_idx in zip(labels, centroid_indexes):
            u, v = centroid_idx[1:3]
            axes[col].annotate(label, (u, v), color="white", size=6)
            axes[col].scatter(u, v, color="white", s=8)

        for pred_label, pred_centroid_idx in zip(pred_labels, pred_centroid_estimates):
            u, v = pred_centroid_idx[1:3]
            axes[col].annotate(pred_label, (u, v), color="red", size=6)
            axes[col].scatter(u, v, color="red", s=8)

        # get average distance
        total_difference = 0.0
        no = 0.0
        for pred_label, pred_centroid_idx in zip(pred_labels, pred_centroid_estimates):
            if pred_label in labels:
                label_idx = labels.index(pred_label)
                print(pred_label, centroid_indexes[label_idx], pred_centroid_idx)
                total_difference += np.linalg.norm(pred_centroid_idx - centroid_indexes[label_idx])
                no += 1

        average_difference = total_difference / no
        print("average", average_difference)
        axes[col].set_xlabel("{:.2f}".format(average_difference) + "mm", fontsize=10)

        i += 1

    fig.subplots_adjust(wspace=0.2, hspace=0.4)
    fig.savefig(plot_path + '/identification-complete.png')


# test_multiple_scans("datasets_test")
# compete_detection_picture('datasets_test', 'saved_current_models', 'plots')
complete_identification_picture('datasets_test', 'saved_current_models/detec-15:59-20e.h5',
                                'saved_current_models/ident-9:59-18e.h5', 'plots',
                                spacing=(1.0, 1.0, 1.0))

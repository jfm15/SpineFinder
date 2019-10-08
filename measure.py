# The aim of this script is to provide measurements for any part of the pipeline
import sys
import glob
import numpy as np
import keras_metrics as km
from utility_functions import opening_files
from keras.models import load_model
from losses_and_metrics.keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy
from keras_models.identification import ignore_background_loss, vertebrae_classification_rate
from losses_and_metrics.dsc import dice_coef_label
from utility_functions.labels import LABELS_NO_L6, VERTEBRAE_SIZES
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
                # output[corner_c[0]:corner_d[0], corner_c[1]:corner_d[1], corner_c[2]:corner_d[2]] = decat_result
                # print(x, y, z, np.bincount(decat_result.reshape(-1).astype(int)))

    return output[border[0]:border[0] + volume.shape[0],
           border[1]:border[1] + volume.shape[1],
           border[2]:border[2] + volume.shape[2]]


def apply_identification_model(volume, i_min, i_max, model):

    paddings = np.mod(16 - np.mod(volume.shape[1:3], 16), 16)
    paddings = np.array(list(zip(np.zeros(3), [0] + list(paddings)))).astype(int)
    volume_padded = np.pad(volume, paddings, mode="constant")
    output = np.zeros(volume_padded.shape)
    i_min = max(i_min, 4)
    i_max = min(i_max, volume_padded.shape[0] - 4)

    for i in range(i_min, i_max, 1):
        volume_slice_padded = volume_padded[i-4:i+4, :, :]
        volume_slice_padded = np.transpose(volume_slice_padded, (1, 2, 0))
        # volume_slice_padded = volume_padded[i, :, :]
        patch = volume_slice_padded.reshape(1, *volume_slice_padded.shape)
        # patch = volume_slice_padded.reshape(1, *volume_slice_padded.shape, 1)
        result = model.predict(patch)
        result = np.squeeze(result, axis=0)
        result = np.squeeze(result, axis=-1)
        result = np.round(result)
        output[i, :, :] = result

    return output[:volume.shape[0], :volume.shape[1], :volume.shape[2]]


def test_scan(scan_path, detection_model, detection_X_shape, detection_y_shape,
              identification_model, spacing=(2.0, 2.0, 2.0)):

    volume = opening_files.read_nii(scan_path, spacing)

    # first stage is to put the volume through the detection model to find where vertebrae are
    print("apply detection")
    detections = apply_detection_model(volume, detection_model, detection_X_shape, detection_y_shape)
    print("finished detection")

    # get the largest island
    # _, largest_island_np = sampling_helper_functions.crop_labelling(detections)
    largest_island_np = np.transpose(np.nonzero(detections))
    # largest_island_np = np.transpose(np.nonzero(largest_island_np)).astype(int)
    i_min = np.min(largest_island_np[:, 0])
    i_max = np.max(largest_island_np[:, 0])

    # second stage is to pass slices of this to the identification network
    print("apply identification")
    identifications = apply_identification_model(volume, i_min, i_max, identification_model)
    print("finished identification")

    # crop parts of slices
    identifications *= detections
    print("finished multiplying")

    # aggregate the predictions
    print("start aggregating")
    identifications = np.round(identifications).astype(int)
    histogram = {}
    for key in range(1, len(LABELS_NO_L6)):
        histogram[key] = np.argwhere(identifications == key)
    '''
    for i in range(identifications.shape[0]):
        for j in range(identifications.shape[1]):
            for k in range(identifications.shape[2]):
                key = identifications[i, j, k]
                if key != 0:
                    if key in histogram:
                        histogram[key] = histogram[key] + [[i, j, k]]
                    else:
                        histogram[key] = [[i, j, k]]
    '''
    print("finish aggregating")

    print("start averages")
    # find averages
    labels = []
    centroid_estimates = []
    for key in sorted(histogram.keys()):
        if 0 <= key < len(LABELS_NO_L6):
            arr = histogram[key]
            # print(LABELS_NO_L6[key], arr.shape[0])
            if arr.shape[0] > max(VERTEBRAE_SIZES[LABELS_NO_L6[key]]**3 * 0.4, 3000):
                print(LABELS_NO_L6[key], arr.shape[0])
                centroid_estimate = np.median(arr, axis=0)
                # ms = MeanShift(bin_seeding=True, min_bin_freq=300)
                # ms.fit(arr)
                # centroid_estimate = ms.cluster_centers_[0]
                centroid_estimate = np.around(centroid_estimate, decimals=2)
                labels.append(LABELS_NO_L6[key])
                centroid_estimates.append(list(centroid_estimate))
    print("finish averages")

    return labels, centroid_estimates, detections, identifications


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


def complete_identification_picture(scans_dir, detection_model_path, identification_model_path, plot_path, start, end,
                                    spacing=(2.0, 2.0, 2.0)):
    scan_paths = glob.glob(scans_dir + "/**/*.nii.gz", recursive=True)[start:end]
    no_of_scan_paths = len(scan_paths)

    weights = np.array([0.1, 0.9])
    detection_model_objects = {'loss': weighted_categorical_crossentropy(weights),
                               'binary_recall': km.binary_recall(),
                               'dice_coef': dice_coef_label(label=1)}

    detection_model = load_model(detection_model_path, custom_objects=detection_model_objects)

    identification_model_objects = {'ignore_background_loss': ignore_background_loss,
                                    'vertebrae_classification_rate': vertebrae_classification_rate}

    identification_model = load_model(identification_model_path, custom_objects=identification_model_objects)

    fig, axes = plt.subplots(nrows=1, ncols=no_of_scan_paths, figsize=(15, 6), dpi=300)

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
        # axes[0].set_ylabel(name, rotation=0, labelpad=50, fontsize=10)

        pred_labels, pred_centroid_estimates, pred_detections, pred_identifications = test_scan(
            scan_path=scan_path,
            detection_model=detection_model,
            detection_X_shape=np.array([64, 64, 80]),
            detection_y_shape=np.array([32, 32, 40]),
            identification_model=identification_model,
            spacing=spacing)

        volume = opening_files.read_nii(scan_path, spacing=spacing)

        volume_slice = volume[cut, :, :]
        # detections_slice = pred_detections[cut, :, :]
        identifications_slice = pred_identifications[cut, :, :]
        # identifications_slice = np.max(pred_identifications, axis=0)

        # masked_data = np.ma.masked_where(identifications_slice == 0, identifications_slice)
        # masked_data = np.ma.masked_where(detections_slice == 0, detections_slice)

        axes[col].imshow(volume_slice.T, cmap='gray', origin='lower')
        # axes[col].imshow(masked_data.T, vmin=1, vmax=27, cmap=cm.jet, alpha=0.4, origin='lower')

        for label, centroid_idx in zip(labels, centroid_indexes):
            u, v = centroid_idx[1:3]
            axes[col].annotate(label, (u, v), color="white", size=6)
            axes[col].scatter(u, v, color="white", s=8)

        axes[col].plot(centroid_indexes[:, 1], centroid_indexes[:, 2], color="white")

        for pred_label, pred_centroid_idx in zip(pred_labels, pred_centroid_estimates):
            u, v = pred_centroid_idx[1:3]
            axes[col].annotate(pred_label, (u, v), color="red", size=6)
            axes[col].scatter(u, v, color="red", s=8)

        pred_centroid_estimates = np.array(pred_centroid_estimates)
        axes[col].plot(pred_centroid_estimates[:, 1], pred_centroid_estimates[:, 2], color="red")

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
    fig.savefig(plot_path + '/centroids_' + str(start) + '_' + str(end) + '.png')


def get_stats(scans_dir, detection_model_path, identification_model_path, spacing=(1.0, 1.0, 1.0)):

    print("detection model: ", detection_model_path)
    print("identification model: ", identification_model_path)
    scan_paths = glob.glob(scans_dir + "/**/*.nii.gz", recursive=True)

    weights = np.array([0.1, 0.9])
    detection_model_objects = {'loss': weighted_categorical_crossentropy(weights),
                     'binary_recall': km.binary_recall(),
                     'dice_coef': dice_coef_label(label=1)}

    detection_model = load_model(detection_model_path, custom_objects=detection_model_objects)

    identification_model_objects = {'ignore_background_loss': ignore_background_loss,
                                    'vertebrae_classification_rate': vertebrae_classification_rate}

    identification_model = load_model(identification_model_path, custom_objects=identification_model_objects)

    all_correct = 0.0
    all_no = 0.0
    cervical_correct = 0.0
    cervical_no = 0.0
    thoracic_correct = 0.0
    thoracic_no = 0.0
    lumbar_correct = 0.0
    lumbar_no = 0.0

    all_difference = []
    cervical_difference = []
    thoracic_difference = []
    lumbar_difference = []

    differences_per_vertebrae = {}

    for i, scan_path in enumerate(scan_paths):
        print(i, scan_path)
        scan_path_without_ext = scan_path[:-len(".nii.gz")]
        centroid_path = scan_path_without_ext + ".lml"

        labels, centroids = opening_files.extract_centroid_info_from_lml(centroid_path)
        centroid_indexes = centroids / np.array(spacing)

        pred_labels, pred_centroid_estimates, pred_detections, pred_identifications = test_scan(
            scan_path=scan_path,
            detection_model=detection_model,
            detection_X_shape=np.array([64, 64, 80]),
            detection_y_shape=np.array([32, 32, 40]),
            identification_model=identification_model,
            spacing=spacing)

        for label, centroid_idx in zip(labels, centroid_indexes):
            min_dist = 20
            min_label = ''
            for pred_label, pred_centroid_idx in zip(pred_labels, pred_centroid_estimates):
                dist = np.linalg.norm(pred_centroid_idx - centroid_idx)
                if dist <= min_dist:
                    min_dist = dist
                    min_label = pred_label

            all_no += 1
            if label[0] == 'C':
                cervical_no += 1
            elif label[0] == 'T':
                thoracic_no += 1
            elif label[0] == 'L':
                lumbar_no += 1

            if label == min_label:
                all_correct += 1
                if label[0] == 'C':
                    cervical_correct += 1
                elif label[0] == 'T':
                    thoracic_correct += 1
                elif label[0] == 'L':
                    lumbar_correct += 1

            print(label, min_label)

        # get average distance
        total_difference = 0.0
        no = 0.0
        for pred_label, pred_centroid_idx in zip(pred_labels, pred_centroid_estimates):
            if pred_label in labels:
                label_idx = labels.index(pred_label)
                print(pred_label, centroid_indexes[label_idx], pred_centroid_idx)
                difference = np.linalg.norm(pred_centroid_idx - centroid_indexes[label_idx])
                total_difference += difference
                no += 1

                # Add to specific vertebrae hash
                if pred_label in differences_per_vertebrae:
                    differences_per_vertebrae[pred_label].append(difference)
                else:
                    differences_per_vertebrae[pred_label] = [difference]

                # Add to total difference
                all_difference.append(difference)
                if pred_label[0] == 'C':
                    cervical_difference.append(difference)
                elif pred_label[0] == 'T':
                    thoracic_difference.append(difference)
                elif pred_label[0] == 'L':
                    lumbar_difference.append(difference)

        average_difference = total_difference / no
        print("average", average_difference, "\n")

    data = []
    labels_used = []
    for label in LABELS_NO_L6:
        if label in differences_per_vertebrae:
            labels_used.append(label)
            data.append(differences_per_vertebrae[label])

    plt.figure(figsize=(20, 10))
    plt.boxplot(data, labels=labels_used)
    plt.savefig('plots/boxplot.png')

    all_rate = np.around(100.0 * all_correct / all_no, decimals=1)
    all_mean = np.around(np.mean(all_difference), decimals=2)
    all_std = np.around(np.std(all_difference), decimals=2)
    cervical_rate = np.around(100.0 * cervical_correct / cervical_no, decimals=1)
    cervical_mean = np.around(np.mean(cervical_difference), decimals=2)
    cervical_std = np.around(np.std(cervical_difference), decimals=2)
    thoracic_rate = np.around(100.0 * thoracic_correct / thoracic_no, decimals=1)
    thoracic_mean = np.around(np.mean(thoracic_difference), decimals=2)
    thoracic_std = np.around(np.std(thoracic_difference), decimals=2)
    lumbar_rate = np.around(100.0 * lumbar_correct / lumbar_no, decimals=1)
    lumbar_mean = np.around(np.mean(lumbar_difference), decimals=2)
    lumbar_std = np.around(np.std(lumbar_difference), decimals=2)

    print("All Id rate: " + str(all_rate) + "%  mean: " + str(all_mean) + "  std: " + str(all_std) + "\n")
    print("Cervical Id rate: " + str(cervical_rate) + "%  mean:" + str(cervical_mean) + "  std:" + str(cervical_std) + "\n")
    print("Thoracic Id rate: " + str(thoracic_rate) + "%  mean:" + str(thoracic_mean) + "  std:" + str(thoracic_std) + "\n")
    print("Lumbar Id rate: " + str(lumbar_rate) + "%  mean:" + str(lumbar_mean) + "  std:" + str(lumbar_std) + "\n")


def single_detection(scan_path, detection_model_path, plot_path, spacing=(1.0, 1.0, 1.0)):
    scan_path_without_ext = scan_path[:-len(".nii.gz")]
    centroid_path = scan_path_without_ext + ".lml"

    labels, centroids = opening_files.extract_centroid_info_from_lml(centroid_path)
    centroid_indexes = centroids / np.array(spacing)

    cut = np.round(np.mean(centroid_indexes[:, 0])).astype(int)

    weights = np.array([0.1, 0.9])
    detection_model_objects = {'loss': weighted_categorical_crossentropy(weights),
                               'binary_recall': km.binary_recall(),
                               'dice_coef': dice_coef_label(label=1)}

    #detection_model = load_model(detection_model_path, custom_objects=detection_model_objects)

    volume = opening_files.read_nii(scan_path, spacing=spacing)

    #detections = apply_detection_model(volume, detection_model, np.array([64, 64, 80]), np.array([32, 32, 40]))

    volume_slice = volume[cut, :, :]
    #detections_slice = detections[cut, :, :]

    # masked_data = np.ma.masked_where(detections_slice == 0, detections_slice)

    fig, ax = plt.subplots(1)

    ax.imshow(volume_slice.T, cmap='gray')
    # ax.imshow(masked_data.T, cmap=cm.jet, alpha=0.4, origin='lower')
    fig.savefig(plot_path + '/single.png')


def single_identification(scan_path, detection_model_path, identification_model_path,
                          plot_path, spacing=(1.0, 1.0, 1.0)):
    scan_path_without_ext = scan_path[:-len(".nii.gz")]
    centroid_path = scan_path_without_ext + ".lml"

    labels, centroids = opening_files.extract_centroid_info_from_lml(centroid_path)
    centroid_indexes = centroids / np.array(spacing)

    cut = np.round(np.mean(centroid_indexes[:, 0])).astype(int)

    weights = np.array([0.1, 0.9])
    detection_model_objects = {'loss': weighted_categorical_crossentropy(weights),
                               'binary_recall': km.binary_recall(),
                               'dice_coef': dice_coef_label(label=1)}

    detection_model = load_model(detection_model_path, custom_objects=detection_model_objects)

    identification_model_objects = {'ignore_background_loss': ignore_background_loss,
                                    'vertebrae_classification_rate': vertebrae_classification_rate}

    identification_model = load_model(identification_model_path, custom_objects=identification_model_objects)

    volume = opening_files.read_nii(scan_path, spacing=spacing)

    detections = apply_detection_model(volume, detection_model, np.array([64, 64, 80]), np.array([32, 32, 40]))
    identification = apply_identification_model(volume, cut - 1, cut + 1, identification_model)

    volume_slice = volume[cut, :, :]
    detection_slice = detections[cut, :, :]
    identification_slice = identification[cut, :, :]

    identification_slice *= detection_slice

    masked_data = np.ma.masked_where(identification_slice == 0, identification_slice)

    fig, ax = plt.subplots(1)

    ax.imshow(volume_slice.T, cmap='gray')
    ax.imshow(masked_data.T, cmap=cm.jet, vmin=1, vmax=27, alpha=0.4, origin='lower')
    fig.savefig(plot_path + '/single_identification.png')


get_stats(sys.argv[1], sys.argv[2], sys.argv[3], spacing=(1.0, 1.0, 1.0))



# The aim of this script is to get the final centroid positions from a scan
import numpy as np
import keras_metrics as km
from utility_functions import opening_files, sampling_helper_functions
from keras.models import load_model
from losses_and_metrics.keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy
from models.six_conv_slices import cool_loss


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


def apply_identification_model(volume, bounds, model):
    i_min, i_max, j_min, j_max, k_min, k_max = bounds
    cropped_volume = volume[i_min:i_max, j_min:j_max, k_min:k_max]
    print(cropped_volume.shape)
    output = np.zeros(volume.shape)

    for i in range(i_max - i_min):
        volume_slice = cropped_volume[i, :, :]
        volume_slice = volume_slice.reshape(1, *volume_slice.shape, 1)
        prediction = model.predict(volume_slice)
        prediction = prediction.reshape(*volume_slice.shape)
        output[i, j_min:j_max, k_min:k_max] = prediction

    return output


def test_individual_scan(scan_path,
                         detection_model_path, detection_model_input_shape, detection_model_objects,
                         identification_model_path, identification_model_objects):

    volume = opening_files.read_nii(scan_path)

    # first stage is to put the volume through the detection model to find where vertebrae are
    detection_model = load_model(detection_model_path, custom_objects=detection_model_objects)
    detections = apply_detection_model(volume, detection_model, detection_model_input_shape)

    # get the largest island
    bounds, detections = sampling_helper_functions.crop_labelling(detections)

    # second stage is to pass slices of this to the identification network
    identification_model = load_model(identification_model_path, custom_objects=identification_model_objects)
    identifications = apply_identification_model(volume, bounds, identification_model)

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

    label_translation = ["B", "C1", "C2", "C3", "C4", "C5", "C6", "C7",
                         "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
                         "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "L6",
                         "S1", "S2"]

    # find averages
    for key in histogram.keys():
        arr = np.array(histogram[key])
        if arr.shape[0] > 100:
            x, y, z = np.mean(arr, axis=0)
            print(label_translation[key], x, y, z)


# modes
# print identification_map
weights = np.array([0.1, 0.9])
detection_model_objects = {'loss': weighted_categorical_crossentropy(weights),
                         'binary_recall': km.binary_recall()}
identification_model_objects = {'cool_loss': cool_loss}
test_individual_scan(scan_path="datasets/spine-1/patient0088/2684937/2684937.nii.gz",
                     detection_model_path="model_files/two_class_model.h5",
                     detection_model_input_shape=np.array([28, 28, 28]),
                     detection_model_objects=detection_model_objects,
                     identification_model_path="model_files/slices_model.h5",
                     identification_model_objects=identification_model_objects)
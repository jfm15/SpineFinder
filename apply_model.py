import numpy as np
import keras_metrics as km
from keras.models import load_model
from losses_and_metrics.keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy
from utility_functions import opening_files


def apply_model(data_path, model_dir, prediction_dir, patch_size, custom_objects={}, file_ext=".nii.gz"):

    volume = opening_files.read_nii(data_path)

    model = load_model(model_dir, custom_objects=custom_objects)

    ext_len = len(file_ext)
    name = (data_path.rsplit('/', 1)[-1])[:-ext_len]
    name = name + "-prediction"

    output = np.zeros(volume.shape)

    for x in range(0, volume.shape[0] - patch_size[0], patch_size[0]):
        for y in range(0, volume.shape[1] - patch_size[1], patch_size[1]):
            for z in range(0, volume.shape[2] - patch_size[2], patch_size[2]):
                corner_a = [x, y, z]
                corner_b = corner_a + patch_size
                patch = volume[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]
                patch = np.expand_dims(patch, axis=0)
                patch = np.expand_dims(patch, axis=4)
                result = model.predict(patch)
                result = np.squeeze(result, axis=0)
                decat_result = np.argmax(result, axis=3)
                output[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]] = decat_result
                print(x, y, z, np.bincount(decat_result.reshape(-1).astype(int)))

    prediction_path = '/'.join([prediction_dir, name])
    np.save(prediction_path, output)


weights = np.array([0.1, 0.9])
recall_background = km.binary_recall(label=0)
recall_vertebrae = km.binary_recall(label=1)
apply_model(data_path="datasets/spine-1/patient0088/2684937/2684937.nii.gz",
            model_dir="model_files/six_conv_20_epochs.h5",
            prediction_dir="predictions/six_conv_20_epochs",
            patch_size=np.array([28, 28, 28]),
            custom_objects={'loss': weighted_categorical_crossentropy(weights),
                            'binary_recall': km.binary_recall()})

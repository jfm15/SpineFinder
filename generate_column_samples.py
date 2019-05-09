import glob
import numpy as np
import keras_metrics as km
from utility_functions import opening_files, sampling_helper_functions
from keras.models import load_model
from losses_and_metrics.keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy
from apply_model import apply_model


def generate_column_samples(dataset_dir, sample_dir, model_path, custom_objects={}, file_ext=".nii.gz"):

    ext_len = len(file_ext)
    for data_path in glob.glob(dataset_dir + "/**/*" + file_ext, recursive=True):

        # get path to corresponding metadata
        data_path_without_ext = data_path[:-ext_len]
        metadata_path = data_path_without_ext + ".lml"

        volume = opening_files.read_nii(data_path)
        model = load_model(model_path, custom_objects=custom_objects)

        prediction = apply_model(volume, model, np.array([28, 28, 28]))
        # print(prediction.shape)
        cropped_prediction = sampling_helper_functions.crop_labelling(prediction)
        # print(cropped_prediction.shape)


weights = np.array([0.1, 0.9])
generate_column_samples(dataset_dir="datasets/spine-1",
                        sample_dir="samples/two_class_cropped",
                        model_path="model_files/two_class_model.h5",
                        custom_objects={'loss': weighted_categorical_crossentropy(weights),
                                        'binary_recall': km.binary_recall()}
                        )
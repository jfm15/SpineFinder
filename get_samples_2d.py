import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utility_functions import dense_labelling as dl, opening_files as of
from utility_functions import helper_functions as hf
import os
import tensorflow as tf
from sample_architecture import cnn_model_fn

"""DEFINE CONSTANTS"""
scales = np.array([0.3125, 0.3125, 2.5])
# block size in mm
# block_size = np.array([96, 112, 32])
block_size = np.array([112, 32])
radii = 13

training_data = np.empty(np.append(0, block_size))
training_data_labels = np.empty(0)

x = 0
dir = './spine-1'
for patient_file in next(os.walk(dir))[1]:
    for scan_number in next(os.walk(dir + '/' + patient_file))[1]:
        print(patient_file)
        if patient_file == 'patient0030':
            continue
        full_dir = dir + '/' + patient_file + '/' + scan_number + '/' + scan_number
        scan = of.read_nii(full_dir + '.nii')
        labels, centroids = of.extract_centroid_info_from_lml(full_dir + '.lml')

        """SLICE SCAN AND LABELLING"""
        scaled_centroids = centroids / scales
        best_transverse_cut = int(round(np.mean(scaled_centroids[:, 0])))

        scan_slice = scan[best_transverse_cut, :, :]

        random_range = scan_slice.shape - block_size

        """GET DENSE LABELLING"""
        dense_labelling = dl.generate_dense_labelling_2D(scan, best_transverse_cut, centroids, radii, scales)

        for i in range(10):
            point_1 = np.round(np.random.rand(2) * random_range).astype(int)
            point_2 = point_1 + block_size
            test_point = (point_1 + block_size/2).astype(int)
            part_of_vertebrae = dense_labelling[test_point[0], test_point[1]] > 0
            sample = scan_slice[point_1[0]:point_2[0], point_1[1]:point_2[1]]
            training_data = np.append(training_data, [sample], axis=0)
            training_data_labels = np.append(training_data_labels, part_of_vertebrae)
            if part_of_vertebrae:
                x += 1
                print(x)

training_data_labels = training_data_labels.astype(int)

# get some more information
tf.logging.set_verbosity(tf.logging.INFO)

# model_dir is where the model data / checkpoints will be saved
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/samples_convnet_model")

# Set up logging for predictions (because it takes time to train)
# Tensors we want to log
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

# Note this is a function passed to the train method below
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": training_data},
    y=training_data_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

print(training_data.shape, training_data_labels.shape)

# train one step and display the probabilities
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=10)


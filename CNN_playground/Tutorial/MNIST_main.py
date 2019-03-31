from __future__ import absolute_import, division, print_function

from MNIST_architecture import cnn_model_fn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# get some more information
tf.logging.set_verbosity(tf.logging.INFO)

# Load training and eval data
((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data/np.float32(255)
train_labels = train_labels.astype(np.int32)  # not required

eval_data = eval_data/np.float32(255)
eval_labels = eval_labels.astype(np.int32)  # not required

# plt.imshow(eval_data[0])
# plt.show()

# model_dir is where the model data / checkpoints will be saved
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

# Set up logging for predictions (because it takes time to train)
# Tensors we want to log
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

# Note this is a function passed to the train method below
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

# train one step and display the probabilities
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=1000)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

print(eval_results)
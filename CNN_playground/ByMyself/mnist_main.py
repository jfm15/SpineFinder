import tensorflow as tf
from mnist_architecture import mnist_model

# Load training and eval data
((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

print (train_data.shape, train_labels.shape, eval_data.shape, eval_labels.shape)

estimator = tf.estimator.Estimator(model_fn=mnist_model, model_dir="james_convnet_model")

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=train_data,
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=eval_data[1:2],
    y=eval_labels[1:2],
    num_epochs=1,
    shuffle=False)

eval_results = estimator.evaluate(input_fn=eval_input_fn)

print(eval_results)
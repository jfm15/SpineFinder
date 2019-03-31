from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

# get some more information
tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):

    # batch size is -1 so that it changes based on the size of features
    # 28, 28 and 1 are the width, height and channels respectively
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # padding="same" adds 0s around the edge of the input to preserve size
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # output has size of [batch_size, 28, 28, 32]

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # output has size of [batch_size, 14, 14, 32]

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # output has size of [batch_size, 14, 14, 64]

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # output has size of [batch_size, 7, 7, 64]

    # flatten layer so we can put it into our fully connected layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # output has shape of [batch_size, 3136]

    # the fully connected layer has 1024 nodes
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # to improve results we have added a dropout layer (40% dropout each time)
    # 3rd argument says that dropout only happens when training = True
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # output has shape [batch_size, 1024]

    # 10 classes corresponds to 10 digits, no activation function
    logits = tf.layers.dense(inputs=dropout, units=10)
    # output has shape [batch_size, 10]

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Predict mode ends here with predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # For training and testing we require loss
    # labels contains [1..9]
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
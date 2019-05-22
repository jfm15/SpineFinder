import numpy as np
from perform_learning import perform_learning
from models.six_conv_multi_classes import six_conv_multi_classes
from models.simple_detection import simple_detection, detection_unet
from models.unet import unet

# inputs to the model
model_params = {'kernel_size': (3, 3, 3),
                'filters': 16,
                'weights': np.array([0.1, 0.9]),
                'learning_rate': 0.001}

perform_learning(sample_dir="samples/two_class",
                 training_val_split=0.5,
                 sample_shape=(64, 64, 80),
                 batch_size=16,
                 sample_channels=1,
                 categorise=True,
                 output_classes=2,
                 model_func=detection_unet,
                 model_params=model_params,
                 epochs=150,
                 model_path="two_class_model.h5",
                 checkpoint_path="checkpoints/two_class_model/two_class_model.{epoch:02d}.h5",
                 log_name="detection",
                 log_description="Using 20 Samples with 2 all zero samples with NO ROTATE and NORMALIZE, using ADAM compiler, "
               "using ReLu and softmax activation, batch norm with mom=0.1")

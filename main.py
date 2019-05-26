import numpy as np
from perform_learning import perform_learning
from models.simple_detection import detection_unet_no_padding

# inputs to the model
model_params = {'kernel_size': (3, 3, 3),
                'filters': 16,
                'weights': np.array([0.1, 0.9]),
                'learning_rate': 0.001}

perform_learning(sample_dir="samples/two_class",
                 training_val_split=0.5,
                 X_shape=(68, 68, 84),
                 y_shape=(28, 28, 44),
                 batch_size=16,
                 sample_channels=1,
                 categorise=True,
                 output_classes=2,
                 model_func=detection_unet_no_padding,
                 model_params=model_params,
                 epochs=150,
                 model_path="two_class_model.h5",
                 checkpoint_path="checkpoints/two_class_model/two_class_model.{epoch:02d}.h5",
                 log_name="detection",
                 log_description="Using 160 Samples with 10 all zero samples with NO ROTATE and NORMALIZE, using ADAM compiler, "
               "using ReLu and softmax activation, batch norm with mom=0.1")

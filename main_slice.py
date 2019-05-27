from perform_learning import perform_learning
from models.simple_identification import simple_identification, unet_slices_no_padding

# inputs to the model
model_params = {'kernel_size': (3, 3),
                'filters': 16,
                'learning_rate': 0.001}

perform_learning(sample_dir="samples/slices",
                 training_val_split=0.5,
                 X_shape=(124, 332),
                 y_shape=(36, 244),
                 batch_size=32,
                 sample_channels=1,
                 categorise=False,
                 output_classes=1,
                 model_func=unet_slices_no_padding,
                 model_params=model_params,
                 epochs=150,
                 model_path="slices_model.h5",
                 checkpoint_path="checkpoints/slices_model/slices_model.{epoch:02d}.h5",
                 log_name="identification",
                 log_description="Using 100 NORMALIZED deformed samples with sigma=7 each with > 500 vertebrae pixels, using ADAM compiler, "
                                 "using ReLu activation")
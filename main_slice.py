from perform_learning import perform_learning
from models.simple_identification import simple_identification, unet_slices

# inputs to the model
model_params = {'kernel_size': (5, 5),
                'filters': 64,
                'learning_rate': 0.001}

perform_learning(sample_dir="samples/slices",
                 training_val_split=0.5,
                 sample_shape=(40, 160),
                 batch_size=64,
                 sample_channels=1,
                 categorise=False,
                 output_classes=1,
                 model_func=unet_slices,
                 model_params=model_params,
                 epochs=150,
                 model_path="slices_model.h5",
                 checkpoint_path="checkpoints/slices_model/slices_model.{epoch:02d}.h5",
                 log_name="identification",
                 log_description="Using 20 NORMALIZED samples each with > 2 vertebrae, using ADAM compiler, "
                                 "using ReLu activation")
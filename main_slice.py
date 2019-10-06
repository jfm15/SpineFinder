from perform_learning import perform_learning
from models.identification import simple_identification, unet_slices

# inputs to the model
model_params = {'kernel_size': (3, 3),
                'filters': 32,
                'learning_rate': 0.001}

perform_learning(training_sample_dir="samples/slices/training",
                 val_sample_dir="samples/slices/testing",
                 batch_size=32,
                 sample_channels=8,
                 categorise=False,
                 output_classes=1,
                 shuffle=True,
                 model_func=unet_slices,
                 model_params=model_params,
                 epochs=35,
                 model_path="slices_model.h5",
                 checkpoint_path="checkpoints/slices_model/slices_model.{epoch:02d}.h5",
                 log_name="identification",
                 log_description="Using 50 samples with sigma=7 each with > 500 vertebrae pixels, using ADAM compiler, "
                                 "using ReLu activation")

from perform_learning import perform_learning
from models.simple_identification import simple_identification, unet_slices

# inputs to the model
model_params = {'kernel_size': (3, 3),
                'filters': 16,
                'learning_rate': 0.001}

perform_learning(sample_dir="samples/slices",
                 batch_size=8,
                 sample_channels=1,
                 categorise=False,
                 output_classes=1,
                 shuffle=True,
                 model_func=unet_slices,
                 model_params=model_params,
                 epochs=1000,
                 model_path="slices_model.h5",
                 checkpoint_path="checkpoints/slices_model/slices_model.{epoch:02d}.h5",
                 log_name="identification",
                 log_description="Using 100 NORMALIZED deformed samples with sigma=7 each with > 500 vertebrae pixels, using ADAM compiler, "
                                 "using ReLu activation")

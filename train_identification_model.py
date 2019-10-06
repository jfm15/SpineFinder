from learning_functions.perform_learning import perform_learning
from keras_models.identification import identification_unet

# inputs to the model
model_params = {'kernel_size': (3, 3),
                'filters': 32,
                'learning_rate': 0.001}

perform_learning(training_sample_dir="samples/slices/training",
                 val_sample_dir="samples/slices/testing",
                 batch_size=32,
                 three_d=False,
                 sample_channels=8,
                 categorise=False,
                 output_classes=1,
                 shuffle=True,
                 model_func=identification_unet,
                 model_params=model_params,
                 epochs=35,
                 model_path="slices_model.h5",
                 checkpoint_path="checkpoints/slices_model/slices_model.{epoch:02d}.h5",
                 log_name="identification",
                 log_description="Using 50 samples with sigma=7 each with > 500 vertebrae pixels, using ADAM compiler, "
                                 "using ReLu activation")

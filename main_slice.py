from perform_learning import perform_learning
from models.simple_identification import simple_identification

# inputs to the model
model_params = {'kernel_size': (5, 5),
                'filters': 32,
                'learning_rate': 0.005}

perform_learning(sample_dir="samples/slices",
                 training_val_split=0.5,
                 sample_shape=(40, 160),
                 batch_size=32,
                 sample_channels=1,
                 categorise=False,
                 output_classes=1,
                 model_func=simple_identification,
                 model_params=model_params,
                 epochs=150,
                 model_path="slices_model.h5",
                 checkpoint_path="checkpoints/slices_model/slices_model.{epoch:02d}.h5",
                 log_name="identification",
                 log_description="Using 20 NORMALIZED samples each with > 2 vertebrae, using ADAM compiler, "
                                 "using ReLu activation")
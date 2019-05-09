from perform_learning import perform_learning
from models.six_conv_slices import six_conv_slices

model = six_conv_slices(kernel_size=(5, 5))

perform_learning(sample_dir="samples/slices",
                 training_val_split=0.8,
                 sample_shape=(None, None, None),
                 batch_size=1,
                 sample_channels=1,
                 categorise=False,
                 output_classes=1,
                 model=model,
                 epochs=20,
                 model_path="slices_model.h5")
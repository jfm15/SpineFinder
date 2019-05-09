import numpy as np
from perform_learning import perform_learning
from models.six_conv_multi_classes import six_conv_multi_classes
from models.six_conv_two_classes import six_conv_two_classes
from models.unet import unet

'''
model = unet(input_shape=(28, 28, 28),
             kernel_size=(3, 3, 3),
             weights=np.array([0.7, 0.93]),
             learning_rate=0.003)
'''


model = six_conv_two_classes(input_shape=(40, 40, 40, 1),
                             kernel_size=(5, 5, 5),
                             weights=np.array([0.1, 0.9]))

'''
weights = np.ones(28) * 0.0384
weights[0] = 0.001
model = six_conv_multi_classes(input_shape=(28, 28, 28, 1),
                               kernel_size=(5, 5, 5),
                               classes=28,
                               weights=weights,
                               learning_rate=0.1)
'''

'''
for layer in model.layers:
    print(layer.name)
    print(layer.input_shape)
    print(layer.output_shape)
'''

perform_learning(sample_dir="samples/multi_class",
                 training_val_split=0.8,
                 sample_shape=(40, 40, 40),
                 batch_size=32,
                 sample_channels=1,
                 output_classes=28,
                 model=model,
                 epochs=20,
                 model_path="main-model.h5")

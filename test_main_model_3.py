from packages.UnetCNN.unet3d import training
import numpy as np

path = 'samples/2551924-34'
sample = np.load(path + '-sample.npy')
sample_labelling = np.load(path + '-labelling.npy')

model = training.load_old_model('main-model.h5')

result = model.predict(sample.reshape(1, 28, 28, 28, 1))
result = result.reshape(28, 28, 28, 27)
print(np.unique(result))
decat_result = np.argmax(result, axis=3)

print(np.bincount(sample_labelling.reshape(-1).astype(int)))
print(np.bincount(decat_result.reshape(-1).astype(int)))

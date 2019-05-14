from create_partition import create_partition_and_labels
from data_generator import DataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from time import time


def perform_learning(sample_dir, training_val_split, sample_shape,
                     batch_size, sample_channels, categorise, output_classes,
                     model, epochs, model_path, checkpoint_path):

    # create partition
    partition, labels = create_partition_and_labels(sample_dir, training_val_split, randomise=True)

    # generators
    params = {'dim': sample_shape,
              'samples_dir': sample_dir,
              'batch_size': batch_size,
              'n_channels': sample_channels,
              'categorise': categorise,
              'n_classes': output_classes,
              'shuffle': True}

    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)

    # set checkpoint
    checkpoint = ModelCheckpoint(checkpoint_path, period=5)

    # tensorboard
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    # train the model
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6,
                        epochs=epochs,
                        callbacks=[checkpoint, tensorboard])

    model.save(model_path)


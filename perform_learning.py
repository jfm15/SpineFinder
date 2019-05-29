from create_partition import create_partition_and_labels
from data_generator import DataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import os
import inspect
import gc


def perform_learning(sample_dir, training_val_split,
                     batch_size, sample_channels, categorise, output_classes, shuffle,
                     model_func, model_params, epochs, model_path, checkpoint_path,
                     log_name, log_description):

    # create partition
    partition, labels = create_partition_and_labels(sample_dir, training_val_split, randomise=False)

    print(partition['validation'])

    # generators
    params = {'samples_dir': sample_dir,
              'batch_size': batch_size,
              'n_channels': sample_channels,
              'categorise': categorise,
              'n_classes': output_classes,
              'shuffle': shuffle}

    training_generator = DataGenerator(partition['validation'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)

    # set checkpoint
    checkpoint = ModelCheckpoint(checkpoint_path, period=3)

    # create model
    model = model_func(**model_params)

    # tensorboard
    now = datetime.datetime.now()
    tensorboard_name = now.strftime("%Y-%m-%d-%H:%M")
    tensorboard_name = log_name + '-' + tensorboard_name
    path = "logs/" + tensorboard_name
    tensorboard = TensorBoard(log_dir=path)

    # create description file
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = path + "/description.txt"

    # get arguments to this function call
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)

    file = open(file_path, 'w')
    for i in args:
        file.write("%s = %s \n" % (i, values[i]))

    # add other info
    file.write("\n\nOTHER INFO\n")
    file.write(log_description)
    file.write("\n\n")

    # https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
    with file as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    file.close()

    # train the mode
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6,
                        epochs=epochs,
                        callbacks=[checkpoint, tensorboard])

    model.save(model_path)

    del model
    gc.collect()


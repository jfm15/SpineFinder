## Introduction

## Setup

Note: Shortly this setup guide will be replaced with a setup.py

The experiments for the paper were run on a microsoft azure vm (https://azuremarketplace.microsoft.com/en-gb/marketplace/apps/microsoft-dsvm.linux-data-science-vm-ubuntu) with NC6 Promo.
Once 'ssh'ed into the vm the following tasks were performed:

1. Clone this repository
1. We used conda to install the packages required for this project:<br>
`conda create -n spine-env`<br>
`source activate spine-env`<br>
`conda install tensorflow-gpu==1.12.0`<br>
`conda install keras`<br>
`conda install matplotlib`<br>
`conda install -c https://conda.anaconda.org/simpleitk SimpleITK`<br>
`conda install pip`<br>
`pip install keras-metrics`<br>
`pip install elasticdeform`<br>

## Usage
To reproduce the results of the paper follow these instructions:
1. First you must download the data from BioMedia: https://biomedia.doc.ic.ac.uk/data/spine/. 
In the dropbox package there are collections of spine scans called 'spine-1', 'spine-2', 'spine-3', 
'spine-4' and 'spine-5', download and unzip these files and move all these scans into a directory called
'training_dataset'. You will also see a zip file called 'spine-test-data', download and unzip this file 
and rename it 'testing_dataset'.
1. You must then generate samples to train and test the detection network. 
`python generate_detection_samples.py 'training_dataset' 'samples/detection/training'`
`python generate_detection_samples.py 'testing_dataset' 'samples/detection/testing'`
1. Now train the detection network: `python train_detection_model.py 'samples/detection/training' 'samples/detection/testing' 'saved_models/detection.h5'`
1. You must then generate samples to train and test the identification network. 
`python generate_identification_samples.py 'training_dataset' 'samples/identification/training'`
`python generate_identification_samples.py 'testing_dataset' 'samples/identification/testing'`
1. Now train the identification network: `python train_identification_model.py 'samples/identification/training' 'samples/identification/testing' 'saved_models/identification.h5'`
1. You can now run the full algorithm on the test data. It should be noted that due to the randomness of sample generation
and the stochastic nature of training a network the results may not be exactly as stated in the paper (could be higher or lower).
To get results for the same metrics run `python measure.py 'testing_dataset' 'saved_models/detection.h5' 'saved_models/identification.h5'` 

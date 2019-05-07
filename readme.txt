To run on lab computers

source /vol/cuda/9.0.176/setup.sh && export CUDA_VISIBLE_DEVICES=0

ssh into the graphics machine by doing
ssh jfm15@shell1.doc.ic.ac.uk
ssh jfm15@graphic10.doc.ic.ac.uk

Useful commands:
ssh jfm15@spine-finder.westeurope.cloudapp.azure.com
scp -r jfm15@spine-finder.westeurope.cloudapp.azure.com:SpineFinder/main-model.h5 /Users/James/SpineFinder
nohup python main.py &> main-output.txt &
(top and kill)

File Structure
|- main.py
|-
|- datasets
|   |- spine-1
|   |   |- patient0001
|   |   |   |- 2804506
|   |   |   |   |- 2804506.lml
|   |   |   |   |- 2804506.nii.gz
|   |   | etc...
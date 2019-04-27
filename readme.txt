To run on lab computers

source /vol/cuda/9.0.176/setup.sh && export CUDA_VISIBLE_DEVICES=0

ssh into the graphics machine by doing
ssh jfm15@shell1.doc.ic.ac.uk
ssh jfm15@graphic10.doc.ic.ac.uk

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
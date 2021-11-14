# spatial-transformer-experiments


## conda environment install 

Using the provided YAML file:

    conda env create --file stn.cpu.yml


 ## conda environment install (alternative install)

    conda create --name stn.cpu
    conda activate stn.cpu
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    conda install matplotlib

## reproducing baseline 

Claimed results (https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html):

    Test set: Average loss: 0.0375, Accuracy: 9880/10000 (99%)

Reproduced Results (3 runs):

    Test set: Average loss: 0.0513, Accuracy: 9852/10000 (99%)
    Test set: Average loss: 0.0486, Accuracy: 9854/10000 (99%)
    Test set: Average loss: 0.0492, Accuracy: 9876/10000 (99%)

Reorganizing Code (rerun baseline using):

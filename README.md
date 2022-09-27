# Neural Transient Field(NeTF)
This repository is based on [torch-ngp] https://github.com/ashawkey/torch-ngp and [NeTF] https://github.com/SCI2020/NeTF. 

## Install

### Install with pip
```bash
pip install -r requirements.txt

# (optional) install the tcnn backbone
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Install with conda
```bash
conda env create -f environment.yml
conda activate torch-ngp
```

### Install encodings
```bash
cd *encoder/
pip install .
```

## Repo Org
```
├── configs                                                                                                       
│   ├── ...                                                                                     
│                                                                                           
├── data                                                                                                       
│   ├── ...                                                                                     
│     
├── logs  # experiment logs                                                                                                                                                                                               
│   ├── experiment 1                                                                                                  
│   │   └── result # reconstructed volume                                                                                                                             
│   │   └── histogram # rendered histogram                                                                                  
│   │   └── Image   # rendered image
|   |   └── model # saved model
|   ├── experiment 2
|   |   └── ...
```
## Train

```
CUDA_VISIBLE_DEVICES=0 python run_netf.py --config configs/xxx
```

## Args
```
expname = run_netf_bunny_diffuse    # experiment name
basedir = ./logs/bunny/     # dir to save experiment

datadir = ./data/bunny_diffuse_zaragoza_256.mat     # data dir
dataset_type = nlos     # data type
neglect_zero_bins = True    # if neglect useless bins or not
neglect_former_nums = 100   # cut bins nums from start
neglect_back_nums = 112     # cut bins nums from end

encoding = hashgrid     # encoding type for positions
encoding_dir = sphere_harmonics      # encoding type for directions
num_layers = 2      # layer nums for sigma layer
hidden_dim = 64     # hidden dim for sigma layer
geo_feat_dim = 15   # hidden dim for sigma feature
num_layers_color = 3    # layer nums for color layer
hidden_dim_color = 64   # hidden dim for color layer
bound = 1   # scene boundary
reso = 64   # reconstruction rsolution

N_iters = 1000      # total train iterations
bin_batch= 1024     # num of bins per iteration
lrate = 2e-3        # start lr
lr_decay_rate = 0.1    # lr decay
sampling_points_nums = 37   # num of points per bin

i_loss = 100   # log step for loss
i_hist = 100   # log step for histogram
i_image = 100  # log step for recon image
i_model = 1000  # log step for model
i_print = 1000  # log step for print
```
#!/bin/bash

#SBATCH -p mit_normal_gpu --gres=gpu:h200:1

module load miniforge

# source ~/.conda/envs/image/bin/python
source activate image

python kidney_scan_focused.py
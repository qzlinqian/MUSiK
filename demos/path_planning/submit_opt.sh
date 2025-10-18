#!/bin/bash

#SBATCH -p mit_normal 

module load miniforge

source ~/.conda/envs/image/bin/python

# python kidney_scan_sa.py
python optimize.py

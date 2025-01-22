#!/bin/bash

#SBATCH --gres=gpu:volta:2 --exclusive

# Loading the required module(s)
module load anaconda/Python-ML-2024b

# python kidney_scan_sa.py
python qian_test.py

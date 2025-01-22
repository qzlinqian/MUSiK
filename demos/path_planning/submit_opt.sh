#!/bin/bash

#SBATCH --gres=gpu:volta:2 --exclusive

# python kidney_scan_sa.py
python optimize.py

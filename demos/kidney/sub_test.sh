#!/bin/bash

# #SBATCH -p mit_normal_gpu

#SBATCH -p mit_normal_gpu --gres=gpu:h100:2
#SBATCH -c 32
#SBATCH -t 06:00:00
#SBATCH --mem=88G

module load miniforge

# source ~/.conda/envs/image/bin/python
# source activate cleanenv
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/home/qalin/.conda/envs/cleanenv/lib:$LD_LIBRARY_PATH
module load apptainer/1.1.9   # load modules

# python kidney_scan_sa.py
# python qian_test.py
# singularity exec --nv /home/qalin/python_open3d/ python /home/qalin/ultrasound/MUSiK/demos/kidney/qian_test.py
singularity exec --nv \
  --bind /home/qalin/ultrasound/MUSiK:/workspace \
  /home/qalin/python_open3d/ \
  python /workspace/demos/kidney/qian_test.py

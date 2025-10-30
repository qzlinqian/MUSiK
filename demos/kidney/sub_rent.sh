#!/bin/bash


#SBATCH --nodelist=node3503 --account=rres_acc_qalin_2025-10-23_ami5l1kc --reservation=rres_qalin_2025-10-23_ami5l1kc --qos=rres_qos_qalin_2025-10-23_ami5l1kc
#SBATCH --gres=gpu:l40s:4 -c 32
#SBATCH -t 18:00:00
#SBATCH --mem=400G

module load apptainer/1.1.9   # load modules

# python kidney_scan_sa.py
# python qian_test.py
# singularity exec --nv /home/qalin/python_open3d/ python /home/qalin/ultrasound/MUSiK/demos/kidney/qian_test.py
singularity exec --nv \
  --bind /home/qalin/ultrasound/MUSiK:/workspace \
  /home/qalin/python_open3d/ \
  python /workspace/demos/kidney/qian_test.py
#!/bin/bash

#SBATCH -p download
#SBATCH -o my_download.out-%j

python kidney_scan_focused.py
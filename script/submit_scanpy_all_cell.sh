#!/bin/bash

#SBATCH --partition scavenge
#SBATCH --job-name=scanpy_all_cell_job
#SBATCH --output=./output/scanpy_all_cell.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=20:00:00
#SBATCH --mem 30G


module restore dl_modules
source activate DL_GPU

python -u scanpy_all_cell.py > ./output/scanpy_all_cell_py.txt

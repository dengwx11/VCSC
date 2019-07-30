#!/bin/bash

#SBATCH --partition scavenge
#SBATCH --job-name=single_batch_job
#SBATCH --output=./output/single_batch.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=20:00:00
#SBATCH --mem 10G

module restore dl_modules
source activate DL_GPU
python -u single_batch.py > ./output/single_batch_py.txt

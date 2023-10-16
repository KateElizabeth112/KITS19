#!/bin/bash
#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=6:mem=16gb
#PBS -N data_preprocessing

cd ${PBS_O_WORKDIR}
pwd

# Launch virtual environment
module load anaconda3/personal
source activate nnUNetv2

python3 reformatMetadata.py
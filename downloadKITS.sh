#!/bin/bash
#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -N TotalSegmentator_organ_volumes

cd ${PBS_O_WORKDIR}
cd /rds/general/user/kc2322/ephemeral
pwd

# Launch virtual environment
module load anaconda3/personal
source activate nnUNetv2

git clone https://github.com/neheller/kits19
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
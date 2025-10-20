#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --job-name=test
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=ALL

# Limit OpenMp to 1 thread per task
export OMP_NUM_THREADS=1

#load required modules
module load gnu8
module unload openmpi3/3.1.4

#use anaconda
source $Home/anaconda3/bin/activate

# Run python script
python run_experiments.py

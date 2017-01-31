#!/bin/bash

# This script will request one GPU device and 1 CPU core
#SBATCH --account=pi-evtimov
#SBATCH --job-name=latentSVM
#SBATCH --output=latentSVM.out
#SBATCH --error=latentSVM.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1


# if your executable was built with CUDA, be sure to load the correct CUDA module:
module load python
module unload cuda
module load cuda/7.5

#
# your GPU-based executable here
#

python CNNForMnist_em_svm_rotation.py 


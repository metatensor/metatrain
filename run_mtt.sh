#!/bin/bash
#SBATCH --job-name=qm7-diverse
#SBATCH --output=qm7-diverse.out
#SBATCH --error=qm7-diverse.err
#SBATCH --account=cosmo
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8


module load gcc/13.2.0 python/3.11.7 cuda/12.4.1
conda activate metatrain-pr

mtt train options-mtt.yaml

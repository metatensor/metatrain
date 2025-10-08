#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 2  # must equal to the number of GPUs
#SBATCH --ntasks-per-node 2
#SBATCH --gpus-per-node 2  # use 2 GPUs
#SBATCH --cpus-per-task 8
#SBATCH --exclusive
#SBATCH --partition=h100  # adapt this to your cluster
#SBATCH --time=1:00:00

# load modules and/or virtual environments and/or containers here

srun mtt train options-distributed.yaml
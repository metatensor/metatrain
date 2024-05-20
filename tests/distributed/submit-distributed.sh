#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --ntasks-per-node 2
#SBATCH --gpus-per-node 2
#SBATCH --cpus-per-task 8
#SBATCH --exclusive
#SBATCH --time=1:00:00


# load modules and/or virtual environments and/or containers here

srun metatensor-models train options-distributed.yaml

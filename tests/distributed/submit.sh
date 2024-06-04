#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --time=1:00:00


# load modules and/or virtual environments and/or containers here

metatensor-models train options.yaml

#!/bin/bash
#SBATCH --chdir /home/bigi/metatensor-models/tests/resources/
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --ntasks-per-node 2
#SBATCH --gpus-per-node 2
#SBATCH --cpus-per-task 8
#SBATCH --exclusive
#SBATCH --time=1:00:00


echo STARTING AT `date`

module load gcc python
source /home/bigi/virtualenv-i/bin/activate

srun metatensor-models train options.yaml

echo FINISHED at `date`

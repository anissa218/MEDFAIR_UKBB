#!/bin/bash


echo "------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.7.0
module load GCCcore/12.2.0
module load openmpi/1.10.3

source /well/papiez/users/hri611/python/MEDFAIR-PROJECT/medfair-${MODULE_CPU_TYPE}/bin/activate

python make_pkl_val.py
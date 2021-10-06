#!/bin/bash
#SBATCH --job-name=en_train
#SBATCH --output=/private/home/%u/logs/%x.out
#SBATCH --error=/private/home/%u/logs/%x.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 2
#SBATCH --cpus-per-task 20
#SBATCH --mem-per-cpu=5GB
#SBATCH --constraint=volta32gb
#SBATCH --time=72:00:00
#SBATCH --signal=USR1@300
#SBATCH --open-mode=append
#SBATCH --partition=dev

# shellcheck disable=SC1091
source /private/home/%u/.bashrc
conda deactivate
conda activate vlnce

module purge
module load cuda/10.1
module load cudnn/v7.6.5.32-cuda.10.1
module load NCCL/2.5.6-1-cuda.10.1

printenv | grep SLURM
set -x
srun -u \
python -u run.py \
    --exp-config vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml \
    --run-type train

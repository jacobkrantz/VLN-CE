#!/bin/bash

# ----------------------------------------------------------------------------
# Example script for distributed training of a waypoint model across a single
# compute node of 8 GPUs. Assumes interactive resources are allocated.
# ----------------------------------------------------------------------------

set -x
set -e

# shellcheck disable=SC1091
source /private/home/%u/.bashrc
conda deactivate
conda activate vlnce

unset MASTER_PORT
unset MASTER_ADDR

module purge
module load cuda/10.1
module load cudnn/v7.6.5.32-cuda.10.1
module load NCCL/2.5.6-1-cuda.10.1

python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node 8 \
    run.py \
    --exp-config vlnce_baselines/config/r2r_waypoint/2-wpn-dc.yaml \
    --run-type train

#!/bin/bash
#SBATCH --job-name=train-array
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

CONFIG_LIST="$1"
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$CONFIG_LIST")
filename=$(basename "$CONFIG")

echo "Running preprocessing for $CONFIG"
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/create_datasets.py --config "$CONFIG"

echo "Running training for $CONFIG"
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/training/run_training.py --config "$CONFIG" >logs/"$filename".out 2>logs/"$filename".err
#!/bin/bash
#SBATCH --job-name=bob
#SBATCH --output=logs/bob.out
#SBATCH --error=logs/bob.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/Bob.yaml
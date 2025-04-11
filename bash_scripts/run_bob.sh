#!/bin/bash
#SBATCH --job-name=bob
#SBATCH --output=bob.out
#SBATCH --error=bob.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/Bob.yaml > logs/bob.out 2> logs/bob.err
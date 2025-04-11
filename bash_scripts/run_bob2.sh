#!/bin/bash
#SBATCH --job-name=bob2
#SBATCH --output=bob2.out
#SBATCH --error=bob2.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/Bob2.yaml > logs/bob2.out 2> logs/bob2.err
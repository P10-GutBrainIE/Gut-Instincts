#!/bin/bash
#SBATCH --job-name=biolinkbert-base
#SBATCH --output=biolinkbert-base.out
#SBATCH --error=biolinkbert-base.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/pretrained/biolinkbert-base.yaml

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/pretrained/biolinkbert-base.yaml >logs/biolinkbert-base.out 2>logs/biolinkbert-base.err

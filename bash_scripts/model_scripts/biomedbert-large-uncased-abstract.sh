#!/bin/bash
#SBATCH --job-name=biomedbert-large-uncased-abstract
#SBATCH --output=biomedbert-large-uncased-abstract.out
#SBATCH --error=biomedbert-large-uncased-abstract.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/pretrained/biomedbert-large-uncased-abstract.yaml

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/pretrained/biomedbert-large-uncased-abstract.yaml >logs/biomedbert-large-abstract.out 2>logs/biomedbert-large-abstract.err

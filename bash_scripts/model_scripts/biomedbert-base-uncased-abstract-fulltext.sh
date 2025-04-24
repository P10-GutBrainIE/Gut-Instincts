#!/bin/bash
#SBATCH --job-name=biomedbert-base-uncased-abstract-fulltext
#SBATCH --output=biomedbert-base-uncased-abstract-fulltext.out
#SBATCH --error=biomedbert-base-uncased-abstract-fulltext.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/pretrained/biomedbert-base-uncased-abstract-fulltext.yaml

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/pretrained/biomedbert-base-uncased-abstract-fulltext.yaml >logs/biomedbert-base-abstract-fulltext.out 2>logs/biomedbert-base-abstract-fulltext.err

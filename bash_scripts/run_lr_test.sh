#!/bin/bash
#SBATCH --job-name=LR_finder_test
#SBATCH --output=LR_finder_test.out
#SBATCH --error=LR_finder_test.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/pretrained/LR_finder_test.yaml

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/pretrained/LR_finder_test.yaml >logs/LR_finder_test.out 2>logs/LR_finder_test.err

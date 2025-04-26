#!/bin/bash
#SBATCH --job-name=lr_test
#SBATCH --output=lr_test.out
#SBATCH --error=lr_test.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/lr_test.yaml

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/lr_test.yaml >logs/lr_test.out 2>logs/lr_test.err

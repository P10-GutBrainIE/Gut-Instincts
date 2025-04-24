#!/bin/bash
#SBATCH --job-name=biom-albert-xxlarge-pmc
#SBATCH --output=biom-albert-xxlarge-pmc.out
#SBATCH --error=biom-albert-xxlarge-pmc.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/pretrained/biom-albert-xxlarge-pmc.yaml

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/pretrained/biom-albert-xxlarge-pmc.yaml >logs/biom-albert-xxlarge-pmc.out 2>logs/biom-albert-xxlarge-pmc.err

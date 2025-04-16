#!/bin/bash
#SBATCH --job-name=BioLinkBERT-base
#SBATCH --output=BioLinkBERT-base.out
#SBATCH --error=BioLinkBERT-base.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/BioMed-RoBERTa-base.yaml > logs/BioMed-RoBERTa-base-pre.out 2> logs/BioMed-RoBERTa-base-pre.err
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BioMed-RoBERTa-base.yaml > logs/BioMed-RoBERTa-base.out 2> logs/BioMed-RoBERTa-base.err
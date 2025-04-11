#!/bin/bash
#SBATCH --job-name=ner_ensemble
#SBATCH --output=ner_log.out
#SBATCH --error=ner_log.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

# ---------- BioLinkBERT-large ----------
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BioLinkBert-large_concatenated_data.yaml > logs/biolink-large_concatenated_data.out 2> logs/biolink-large_concatenated_data.err

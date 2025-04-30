#!/bin/bash
#SBATCH --job-name=ner_ensemble
#SBATCH --output=ner_log.out
#SBATCH --error=ner_log.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

# ---------- BioLinkBERT-base ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/test.yaml > logs/biolink-base-pre.out 2> logs/biolink-base-pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/test.yaml > logs/biolink-base-train.out 2> logs/biolink-base-train.err
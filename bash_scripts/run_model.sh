#!/bin/bash
#SBATCH --job-name=ner_ensemble
#SBATCH --output=ner_log.out
#SBATCH --error=ner_log.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

# ---------- BioLinkBERT-large ----------
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BioLinkBert-large.yaml > logs/biolink-large.out 2> logs/biolink-large.err

# ---------- BioLinkBERT-base ----------
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BioLinkBert-base.yaml > logs/biolink-base.out 2> logs/biolink-base.err

# ---------- BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext ----------
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BiomedBert-base-uncased-abstract-fulltext.yaml > logs/BiomedBERT-af.out 2> logs/BiomedBERT-af.err

# ---------- BiomedNLP-BiomedBERT-base-uncased-abstract ----------
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BiomedBert-base-uncased-abstract.yaml > logs/BiomedBERT-a.out 2> logs/BiomedBERT-a.err

# ---------- BiomedELECTRA-base-uncased-abstract ----------
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BiomedElectra-base-uncased-abstract.yaml > logs/BiomedELECTRA.out 2> logs/BiomedELECTRA.err

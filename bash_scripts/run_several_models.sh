#!/bin/bash
#SBATCH --job-name=ner_ensemble
#SBATCH --output=ner_log.out
#SBATCH --error=ner_log.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

# ---------- BioLinkBERT-base ----------
mkdir -p logs/biolinkbert-base

# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/BioLinkBert-base.yaml >logs/biolinkbert-base/pre.out 2>logs/biolinkbert-base/pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BioLinkBert-base.yaml >logs/biolinkbert-base/train.out 2>logs/biolinkbert-base/train.err

# inference
#srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/inference.py --config training_configs/BioLinkBert-base-freeze-24epochs.yaml >logs/biolink-base-freeze-24epochs-inf.out 2>logs/biolink-base-freeze-24epochs-inf.err

# evaluation
#srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/evaluation/evaluation.py --config training_configs/BioLinkBert-base-freeze-24epochs.yaml >logs/biolink-base-freeze-24epochs-eval.out 2>logs/biolink-base-freeze-24epochs-eval.err


# ---------- BioLinkBERT-large ----------
mkdir -p logs/biolinkbert-large

# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/BioLinkBert-large.yaml >logs/biolinkbert-large/pre.out 2>logs/biolinkbert-large/pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BioLinkBert-large.yaml >logs/biolinkbert-large/train.out 2>logs/biolinkbert-large/train.err


# ---------- BiomedBERT-base-uncased-abstract ----------
mkdir -p logs/biomedbert-base-abstract

# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/BiomedBERT-base-uncased-abstract.yaml >logs/biomedbert-base-abstract/pre.out 2>logs/biomedbert-base-abstract/pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BiomedBERT-base-uncased-abstract.yaml >logs/biomedbert-base-abstract/train.out 2>logs/biomedbert-base-abstract/train.err


# ---------- BiomedBERT-base-uncased-abstract-fulltext ----------
mkdir -p logs/biomedbert-base-abstract

# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/BiomedBERT-base-uncased-abstract-fulltext.yaml >logs/biomedbert-base-abstract-fulltext/pre.out 2>logs/biomedbert-base-abstract-fulltext/pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BiomedBERT-base-uncased-abstract-fulltext.yaml >logs/biomedbert-base-abstract-fulltext/train.out 2>logs/biomedbert-base-abstract-fulltext/train.err


# ---------- BiomedBERT-large-uncased-abstract ----------
mkdir -p logs/biomedbert-large-abstract

# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/BiomedBERT-large-uncased-abstract.yaml >logs/biomedbert-large-abstract/pre.out 2>logs/biomedbert-large-abstract/pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BiomedBERT-large-uncased-abstract.yaml >logs/biomedbert-large-abstract/train.out 2>logs/biomedbert-large-abstract/train.err


# ---------- BiomedElectra-base-uncased-abstract ----------
mkdir -p logs/biomedelectra-base-abstract

# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/BiomedELECTRA-base-uncased-abstract.yaml >logs/biomedelectra-base-abstract/pre.out 2>logs/biomedelectra-base-abstract/pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BiomedELECTRA-base-uncased-abstract.yaml >logs/biomedelectra-base-abstract/train.out 2>logs/biomedelectra-base-abstract/train.err


# ---------- BiomedElectra-large-uncased-abstract ----------
mkdir -p logs/biomedelectra-large-abstract

# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/BiomedELECTRA-large-uncased-abstract.yaml >logs/biomedelectra-large-abstract/pre.out 2>logs/biomedelectra-large-abstract/pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BiomedElectra-large-uncased-abstract.yaml >logs/biomedelectra-large-abstract/train.out 2>logs/biomedelectra-large-abstract/train.err


# ---------- BioM-ALBERT-xxlarge-PMC ----------
mkdir -p logs/biom-albert-xxlarge-pmc

# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/BioM-ALBERT-xxlarge-PMC.yaml >logs/biom-albert-xxlarge-pmc/pre.out 2>logs/biom-albert-xxlarge-pmc/pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BioM-ALBERT-xxlarge-PMC.yaml >logs/biom-albert-xxlarge-pmc/train.out 2>logs/biom-albert-xxlarge-pmc/train.err


# ---------- BioM-ALBERT-xxlarge ----------
mkdir -p logs/biom-albert-xxlarge

# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/BioM-ALBERT-xxlarge.yaml >logs/biom-albert-xxlarge/pre.out 2>logs/biom-albert-xxlarge/pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BioM-ALBERT-xxlarge.yaml >logs/biom-albert-xxlarge/train.out 2>logs/biom-albert-xxlarge/train.err


# ---------- BioMEGATRON ----------



# ----------  ----------

#!/bin/bash
#SBATCH --job-name=ner_ensemble
#SBATCH --output=ner_log.out
#SBATCH --error=ner_log.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

# ---------- biolinkbert-base ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/biolinkbert-base.yaml

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/biolinkbert-base.yaml >logs/biolinkbert-base.out 2>logs/biolinkbert-base.err


# ---------- biolinkbert-large ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/biolinkbert-large.yaml

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/biolinkbert-large.yaml >logs/biolinkbert-large.out 2>logs/biolinkbert-large.err


# ---------- biomedbert-base-uncased-abstract ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/biomedbert-base-uncased-abstract.yaml

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/biomedbert-base-uncased-abstract.yaml >logs/biomedbert-base-abstract.out 2>logs/biomedbert-base-abstract.err


# ---------- biomedbert-base-uncased-abstract-fulltext ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/biomedbert-base-uncased-abstract-fulltext.yaml

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/biomedbert-base-uncased-abstract-fulltext.yaml >logs/biomedbert-base-abstract-fulltext.out 2>logs/biomedbert-base-abstract-fulltext.err


# ---------- biomedbert-large-uncased-abstract ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/biomedbert-large-uncased-abstract.yaml

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/biomedbert-large-uncased-abstract.yaml >logs/biomedbert-large-abstract.out 2>logs/biomedbert-large-abstract.err


# ---------- biomedelectra-base-uncased-abstract ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/biomedelectra-base-uncased-abstract.yaml

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/biomedelectra-base-uncased-abstract.yaml >logs/biomedelectra-base-abstract.out 2>logs/biomedelectra-base-abstract.err


# ---------- biomedelectra-large-uncased-abstract ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/biomedelectra-large-uncased-abstract.yaml

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/biomedelectra-large-uncased-abstract.yaml >logs/biomedelectra-large-abstract.out 2>logs/biomedelectra-large-abstract.err


# ---------- biom-albert-xxlarge-pmc ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/biom-albert-xxlarge-pmc.yaml

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/biom-albert-xxlarge-pmc.yaml >logs/biom-albert-xxlarge-pmc.out 2>logs/biom-albert-xxlarge-pmc.err


# ---------- biom-albert-xxlarge ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/biom-albert-xxlarge.yaml

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/biom-albert-xxlarge.yaml >logs/biom-albert-xxlarge.out 2>logs/biom-albert-xxlarge.err

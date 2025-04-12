#!/bin/bash
#SBATCH --job-name=ner_ensemble
#SBATCH --output=ner_log.out
#SBATCH --error=ner_log.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

# ---------- BioLinkBERT-large ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/ensemble_inference/BioLinkBert-large.yaml > logs/biolink-large-pre.out 2> logs/biolink-large-pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/ensemble_inference/BioLinkBert-large.yaml > logs/biolink-large-train.out 2> logs/biolink-large-train.err

# inference
#srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/inference.py --config training_configs/ensemble_inference/BioLinkBert-large.yaml > logs/biolink-large-inf.out 2> logs/biolink-large-inf.err

# evaluation
#srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/evaluation/evaluation.py --config training_configs/ensemble_inference/BioLinkBert-large.yaml > logs/biolink-large-eval.out 2> logs/biolink-large-eval.err

# ---------- BiomedBERT-base-uncased-abstract ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/ensemble_inference/BiomedBert-abstract.yaml > logs/BiomedBERT-abstract-pre.out 2> logs/BiomedBERT-abstract-pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/ensemble_inference/BiomedBert-abstract.yaml > logs/BiomedBERT-abstract-train.out 2> logs/BiomedBERT-abstract-train.err

# inference
#srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/inference.py --config training_configs/ensemble_inference/BiomedBert-abstract.yaml > logs/BiomedBERT-abstract-inf.out 2> logs/BiomedBERT-abstract-inf.err

# evaluation
#srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/evaluation/evaluation.py --config training_configs/ensemble_inference/BiomedBert-abstract.yaml > logs/BiomedBERT-abstract-eval.out 2> logs/BiomedBERT-abstract-eval.err

# ---------- BiomedBERT-base-uncased-abstract-fulltext ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/ensemble_inference/BiomedBert-abstract-fulltext.yaml > logs/BiomedBERT-abstract-fulltext-pre.out 2> logs/BiomedBERT-abstract-fulltext-pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/ensemble_inference/BiomedBert-abstract-fulltext.yaml > logs/BiomedBERT-abstract-fulltext-train.out 2> logs/BiomedBERT-abstract-fulltext-train.err

# inference
#srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/inference.py --config training_configs/ensemble_inference/BiomedBert-abstract-fulltext.yaml > logs/BiomedBERT-abstract-fulltext-inf.out 2> logs/BiomedBERT-abstract-fulltext-inf.err

# evaluation
#srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/evaluation/evaluation.py --config training_configs/ensemble_inference/BiomedBert-abstract-fulltext.yaml > logs/BiomedBERT-abstract-fulltext-eval.out 2> logs/BiomedBERT-abstract-fulltext-eval.err

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
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/freezing_experiment/BioLinkBert-base-freeze-24epochs.yaml > logs/biolink-base-freeze-24epochs-pre.out 2> logs/biolink-base-freeze-24epochs-pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/freezing_experiment/BioLinkBert-base-freeze-24epochs.yaml > logs/biolink-base-freeze-24epochs-train.out 2> logs/biolink-base-freeze-24epochs-train.err

# inference
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/inference.py --config training_configs/freezing_experiment/BioLinkBert-base-freeze-24epochs.yaml > logs/biolink-base-freeze-24epochs-inf.out 2> logs/biolink-base-freeze-24epochs-inf.err

# evaluation
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/evaluation/evaluation.py --config training_configs/freezing_experiment/BioLinkBert-base-freeze-24epochs.yaml > logs/biolink-base-freeze-24epochs-eval.out 2> logs/biolink-base-freeze-24epochs-eval.err

# ---------- BiomedBERT-base-uncased-abstract ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/freezing_experiment/BiomedBERT-base-uncased-abstract-freeze-24epochs.yaml > logs/BiomedBERT-base-uncased-abstract-freeze-24epochs-pre.out 2> logs/BiomedBERT-base-uncased-abstract-freeze-24epochs-pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/freezing_experiment/BiomedBERT-base-uncased-abstract-freeze-24epochs.yaml > logs/BiomedBERT-base-uncased-abstract-freeze-24epochs-train.out 2> logs/BiomedBERT-base-uncased-abstract-freeze-24epochs-train.err

# inference
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/inference.py --config training_configs/freezing_experiment/BiomedBERT-base-uncased-abstract-freeze-24epochs.yaml > logs/BiomedBERT-base-uncased-abstract-freeze-24epochs-inf.out 2> logs/BiomedBERT-base-uncased-abstract-freeze-24epochs-inf.err

# evaluation
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/evaluation/evaluation.py --config training_configs/freezing_experiment/BiomedBERT-base-uncased-abstract-freeze-24epochs.yaml > logs/BiomedBERT-base-uncased-abstract-freeze-24epochs-eval.out 2> logs/BiomedBERT-base-uncased-abstract-freeze-24epochs-eval.err

# ---------- BiomedBERT-large-uncased-abstract ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/freezing_experiment/BiomedBERT-large-uncased-abstract-freeze-24epochs.yaml > logs/BiomedBERT-large-uncased-abstract-freeze-24epochs-pre.out 2> logs/BiomedBERT-large-uncased-abstract-freeze-24epochs-pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/freezing_experiment/BiomedBERT-large-uncased-abstract-freeze-24epochs.yaml > logs/BiomedBERT-large-uncased-abstract-freeze-24epochs-train.out 2> logs/BiomedBERT-large-uncased-abstract-freeze-24epochs-train.err

# inference
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/inference.py --config training_configs/freezing_experiment/BiomedBERT-large-uncased-abstract-freeze-24epochs.yaml > logs/BiomedBERT-large-uncased-abstract-freeze-24epochs-inf.out 2> logs/BiomedBERT-large-uncased-abstract-freeze-24epochs-inf.err

# evaluation
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/evaluation/evaluation.py --config training_configs/freezing_experiment/BiomedBERT-large-uncased-abstract-freeze-24epochs.yaml > logs/BiomedBERT-large-uncased-abstract-freeze-24epochs-eval.out 2> logs/BiomedBERT-large-uncased-abstract-freeze-24epochs-eval.err

# ---------- BiomedBERT-base-uncased-abstract-fulltext ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/freezing_experiment/BiomedBERT-base-uncased-abstract-fulltext-freeze-24epochs.yaml > logs/BiomedBERT-base-uncased-abstract-fulltext-freeze-24epochs-pre.out 2> logs/BiomedBERT-base-uncased-abstract-fulltext-freeze-24epochs-pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/freezing_experiment/BiomedBERT-base-uncased-abstract-fulltext-freeze-24epochs.yaml > logs/BiomedBERT-base-uncased-abstract-fulltext-freeze-24epochs-train.out 2> logs/BiomedBERT-base-uncased-abstract-fulltext-freeze-24epochs-train.err

# inference
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/inference.py --config training_configs/freezing_experiment/BiomedBERT-base-uncased-abstract-fulltext-freeze-24epochs.yaml > logs/BiomedBERT-base-uncased-abstract-fulltext-freeze-24epochs-inf.out 2> logs/BiomedBERT-base-uncased-abstract-fulltext-freeze-24epochs-inf.err

# evaluation
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/evaluation/evaluation.py --config training_configs/freezing_experiment/BiomedBERT-base-uncased-abstract-fulltext-freeze-24epochs.yaml > logs/BiomedBERT-base-uncased-abstract-fulltext-freeze-24epochs-eval.out 2> logs/BiomedBERT-base-uncased-abstract-fulltext-freeze-24epochs-eval.err

# ---------- BiomedELECTRA-base-uncased-abstract ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/freezing_experiment/BiomedELECTRA-base-uncased-abstract-freeze-24epochs.yaml > logs/BiomedELECTRA-base-uncased-abstract-freeze-24epochs-pre.out 2> logs/BiomedELECTRA-base-uncased-abstract-freeze-24epochs-pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/freezing_experiment/BiomedELECTRA-base-uncased-abstract-freeze-24epochs.yaml > logs/BiomedELECTRA-base-uncased-abstract-freeze-24epochs-train.out 2> logs/BiomedELECTRA-base-uncased-abstract-freeze-24epochs-train.err

# inference
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/inference.py --config training_configs/freezing_experiment/BiomedELECTRA-base-uncased-abstract-freeze-24epochs.yaml > logs/BiomedELECTRA-base-uncased-abstract-freeze-24epochs-inf.out 2> logs/BiomedELECTRA-base-uncased-abstract-freeze-24epochs-inf.err

# evaluation
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/evaluation/evaluation.py --config training_configs/freezing_experiment/BiomedELECTRA-base-uncased-abstract-freeze-24epochs.yaml > logs/BiomedELECTRA-base-uncased-abstract-freeze-24epochs-eval.out 2> logs/BiomedELECTRA-base-uncased-abstract-freeze-24epochs-eval.err

# ---------- BiomedELECTRA-large-uncased-abstract ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/freezing_experiment/BiomedELECTRA-large-uncased-abstract-freeze-24epochs.yaml > logs/BiomedELECTRA-large-uncased-abstract-freeze-24epochs-pre.out 2> logs/BiomedELECTRA-large-uncased-abstract-freeze-24epochs-pre.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/freezing_experiment/BiomedELECTRA-large-uncased-abstract-freeze-24epochs.yaml > logs/BiomedELECTRA-large-uncased-abstract-freeze-24epochs-train.out 2> logs/BiomedELECTRA-large-uncased-abstract-freeze-24epochs-train.err

# inference
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/inference.py --config training_configs/freezing_experiment/BiomedELECTRA-large-uncased-abstract-freeze-24epochs.yaml > logs/BiomedELECTRA-large-uncased-abstract-freeze-24epochs-inf.out 2> logs/BiomedELECTRA-large-uncased-abstract-freeze-24epochs-inf.err

# evaluation
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/evaluation/evaluation.py --config training_configs/freezing_experiment/BiomedELECTRA-large-uncased-abstract-freeze-24epochs.yaml > logs/BiomedELECTRA-large-uncased-abstract-freeze-24epochs-eval.out 2> logs/BiomedELECTRA-large-uncased-abstract-freeze-24epochs-eval.err

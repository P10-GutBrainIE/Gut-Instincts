#!/bin/bash
#SBATCH --job-name=ner_ensemble
#SBATCH --output=ner_log.out
#SBATCH --error=ner_log.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

# ---------- BioLinkBERT-large freeze 16/32 epochs ----------
# preprocessing
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/BioLinkBert-large-freeze-16-epochs-out-of-32-epochs.yaml > logs/biolink-large-freeze-16-epochs-out-of-32-epochs.out 2> logs/biolink-large-freeze-16-epochs-out-of-32-epochs.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BioLinkBert-large-freeze-16-epochs-out-of-32-epochs.yaml > logs/biolink-large-freeze-16-epochs-out-of-32-epochs.out 2> logs/biolink-large-freeze-16-epochs-out-of-32-epochs.err

# inference
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/inference.py --config training_configs/BioLinkBert-large-freeze-16-epochs-out-of-32-epochs.yaml > logs/biolink-large-freeze-16-epochs-out-of-32-epochs.out 2> logs/biolink-large-freeze-16-epochs-out-of-32-epochs.err

# evaluation
#srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/evaluation/evaluation.py --config training_configs/BioLinkBert-large-freeze-16-epochs-out-of-32-epochs.yaml > logs/biolink-large-freeze-16-epochs-out-of-32-epochs.out 2> logs/biolink-large-freeze-16-epochs-out-of-32-epochs.err


# ---------- BioLinkBERT-large freeze 20/32 epochs ----------
# preprocessing
#srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/BioLinkBert-large-freeze-16-epochs-out-of-32-epochs.yaml > logs/biolink-large-freeze-16-epochs-out-of-32-epochs.out 2> logs/biolink-large-freeze-16-epochs-out-of-32-epochs.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BioLinkBert-large-freeze-20-epochs-out-of-32-epochs.yaml > logs/biolink-large-freeze-20-epochs-out-of-32-epochs.out 2> logs/biolink-large-freeze-20-epochs-out-of-32-epochs.err

# inference
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/inference.py --config training_configs/BioLinkBert-large-freeze-20-epochs-out-of-32-epochs.yaml > logs/biolink-large-freeze-20-epochs-out-of-32-epochs.out 2> logs/biolink-large-freeze-20-epochs-out-of-32-epochs.err

# evaluation
#srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/evaluation/evaluation.py --config training_configs/BioLinkBert-large-freeze-20-epochs-out-of-32-epochs.yaml > logs/biolink-large-freeze-20-epochs-out-of-32-epochs.out 2> logs/biolink-large-freeze-20-epochs-out-of-32-epochs.err



# ---------- BioLinkBERT-large freeze 24/32 epochs ----------
# preprocessing
#srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config training_configs/BioLinkBert-large-freeze-16-epochs-out-of-32-epochs.yaml > logs/biolink-large-freeze-16-epochs-out-of-32-epochs.out 2> logs/biolink-large-freeze-16-epochs-out-of-32-epochs.err

# training
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BioLinkBert-large-freeze-24-epochs-out-of-32-epochs.yaml > logs/biolink-large-freeze-24-epochs-out-of-32-epochs.out 2> logs/biolink-large-freeze-24-epochs-out-of-32-epochs.err

# inference
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/inference.py --config training_configs/BioLinkBert-large-freeze-24-epochs-out-of-32-epochs.yaml > logs/biolink-large-freeze-24-epochs-out-of-32-epochs.out 2> logs/biolink-large-freeze-24-epochs-out-of-32-epochs.err

# evaluation
#srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/evaluation/evaluation.py --config training_configs/BioLinkBert-large-freeze-24-epochs-out-of-32-epochs.yaml > logs/biolink-large-freeze-24-epochs-out-of-32-epochs.out 2> logs/biolink-large-freeze-24-epochs-out-of-32-epochs.err

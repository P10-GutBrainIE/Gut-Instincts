#!/bin/bash
#SBATCH --job-name=BERTLSTMCRF
#SBATCH --output=BERTLSTMCRF.out
#SBATCH --error=BERTLSTMCRF.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BERTLSTMCRF.yaml > logs/BERTLSTMCRF.out 2> logs/BERTLSTMCRF.err
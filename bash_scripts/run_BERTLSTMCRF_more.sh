#!/bin/bash
#SBATCH --job-name=BERTLSTMCRF_more
#SBATCH --output=BERTLSTMCRF_more.out
#SBATCH --error=BERTLSTMCRF_more.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config training_configs/BERTLSTMCRF_more.yaml > logs/BERTLSTMCRF_more.out 2> logs/BERTLSTMCRF_more.err
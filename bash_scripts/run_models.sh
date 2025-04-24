#!/bin/bash
#SBATCH --job-name=ner_ensemble
#SBATCH --output=ner_log.out
#SBATCH --error=ner_log.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs
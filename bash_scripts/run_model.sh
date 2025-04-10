#!/bin/bash
#SBATCH --job-name=ner_ensemble
#SBATCH --output=ner_log.out
#SBATCH --error=ner_log.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

mkdir -p logs

# ---------- BioLinkBERT ----------
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py 
--config training_configs/BioLinkBert-large.yaml > logs/biolink.out 2> logs/biolink.err



#srun singularity exec --nv --bind ~/test_env:/test_env /ceph/container/python/python_3>  /bin/bash -c "source /test_env/bin/activate && python src/NER/training.py --model_na>  > logs/biolink.out 2> logs/biolink.err
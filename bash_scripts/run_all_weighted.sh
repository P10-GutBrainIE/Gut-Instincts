#!/bin/bash

mkdir -p logs

for script in training_configs/weighted_training2/*.yaml; do
    filename=$(basename "$script")
    echo "Submitting $script"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$filename
#SBATCH --output=$filename.out
#SBATCH --error=$filename.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

# Preprocessing step
echo "Running preprocessing for $script"
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/main.py --config $script

# Training step
echo "Running training for $script"
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/NER/training.py --config $script >logs/$filename.out 2>logs/$filename.err
EOF

done

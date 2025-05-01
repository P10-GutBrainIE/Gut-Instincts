#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <folder_name>"
  exit 1
fi

if [ ! -d "training_configs/$1" ]; then
  echo "Error: Folder 'training_configs/$1' does not exist."
  exit 1
fi

mkdir -p logs/

for script in training_configs/$1/*.yaml; do
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
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/create_datasets.py --config $script

# Training step
echo "Running training for $script"
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/training/training.py --config $script >logs/$filename.out 2>logs/$filename.err
EOF

done

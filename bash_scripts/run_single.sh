#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <relative/path/to/config.yaml>"
  echo "Example: $0 subfolder/my_config.yaml"
  exit 1
fi

CONFIG_PATH="training_configs/$1"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Error: File '$CONFIG_PATH' does not exist."
  exit 1
fi

filename=$(basename "$CONFIG_PATH")
mkdir -p logs/

echo "Submitting $CONFIG_PATH"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$filename
#SBATCH --output=$filename.out
#SBATCH --error=$filename.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

# Preprocessing step
echo "Running preprocessing for $CONFIG_PATH"
srun singularity exec /ceph/container/python/python_3.12.sif python src/preprocessing/create_datasets.py --config $CONFIG_PATH

# Training step
echo "Running training for $CONFIG_PATH"
srun singularity exec --nv /ceph/container/python/python_3.12.sif python src/training/run_training.py --config $CONFIG_PATH >logs/$filename.out 2>logs/$filename.err
EOF

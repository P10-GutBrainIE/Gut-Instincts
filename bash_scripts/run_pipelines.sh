#!/bin/bash

# Check if the src/pipelines folder exists
if [ ! -d "src/pipelines" ]; then
  echo "Error: Folder 'src/pipelines' does not exist."
  exit 1
fi

mkdir -p logs/

# Loop over all Python scripts in src/pipelines
for script in src/pipelines/*.py; do
    filename=$(basename "$script")
    jobname="${filename%.*}"  # Remove the .py extension for the job name
    echo "Submitting $script"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$jobname
#SBATCH --output=logs/${jobname}.out
#SBATCH --error=logs/${jobname}.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1

echo "Running $script"
srun singularity exec --nv /ceph/container/python/python_3.12.sif python $script
EOF

done

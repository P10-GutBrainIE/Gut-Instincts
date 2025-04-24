#!/bin/bash

for script in bash_scripts/model_scripts/*.sh; do
    echo "Submitting $script"
    sbatch $script || { echo "Failed to submit $script"; continue; }
done

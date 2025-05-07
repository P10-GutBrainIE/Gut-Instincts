#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <folder_name>"
  exit 1
fi

CONFIG_DIR="training_configs/$1"

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Error: Folder '$CONFIG_DIR' does not exist."
  exit 1
fi

CONFIGS=($CONFIG_DIR/*.yaml)
NUM_CONFIGS=${#CONFIGS[@]}

if [ "$NUM_CONFIGS" -eq 0 ]; then
  echo "No YAML files found in '$CONFIG_DIR'."
  exit 1
fi

# Write the list of configs to a file for the array job to read
CONFIG_LIST="config_list.txt"
printf "%s\n" "${CONFIGS[@]}" > "$CONFIG_LIST"

sbatch --array=0-$(($NUM_CONFIGS - 1)) run_array_job.sh "$CONFIG_LIST"
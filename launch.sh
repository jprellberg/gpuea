#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00
#SBATCH --partition=long

PYTHONPATH="$PYTHONPATH:$(pwd)" python3.6 -u launch.py "$SLURM_ARRAY_TASK_ID"

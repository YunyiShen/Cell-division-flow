#!/bin/bash

#SBATCH -p mit_normal
#SBATCH --mem=32G

#SBATCH --job-name=myflow
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1       # adjust as needed
#SBATCH --time=12:00:00         # adjust walltime


echo "Running job $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"

module load miniforge/24.3.0-0
conda activate cytofd
python interpolating.py
python visual.py

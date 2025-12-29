#!/bin/bash

#SBATCH -p mit_normal
#SBATCH --mem=2G

#SBATCH --job-name=myflow
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-59            # 30 jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1       # adjust as needed
#SBATCH --time=12:00:00         # adjust walltime
# Load your environment if needed
# module load python/3.13
# source activate cytofd

echo "Running job $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"

module load miniforge/24.3.0-0
conda activate cytofd
python hydrostatic.py --run_id $SLURM_ARRAY_TASK_ID --tmax 600 --dt 0.004 --N 91 --Stokes True

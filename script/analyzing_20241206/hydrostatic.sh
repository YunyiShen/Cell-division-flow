#!/bin/bash

#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -o hydrostatic.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e hydrostatic.out  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=yshen99@mit.edu

module load miniforge/24.3.0-0


conda activate cytofd
python hydrostatic.py

#!/bin/sh
module load anaconda/2023a-pytorch
conda activate torch
python denoising.py

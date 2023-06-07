#!/bin/bash

#SBATCH -p gpu
#SBATCH --mem=128G

python3 test_sam.py
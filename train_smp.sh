#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti
#SBATCH --mem-per-cpu=16G

python3 train_smp.py --mode "train"
#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti

python3 train_smp.py --mode "predict"
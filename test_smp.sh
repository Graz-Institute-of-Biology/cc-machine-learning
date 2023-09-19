#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1

python3 train_smp.py --mode "test"
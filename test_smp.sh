#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:a100.40gb

python3 train_smp.py --mode "test"
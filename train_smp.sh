#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:a100.40gb:1
#SBATCH --mem-per-cpu=16G

python3 train_smp.py --mode "train"
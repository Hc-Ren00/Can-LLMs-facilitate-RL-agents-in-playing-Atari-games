#!/bin/bash
#SBATCH -c 2 # Number of Cores per Task
#SBATCH --mem=40G # Requested Memory
#SBATCH -p gpu-long # Partition
#SBATCH -G 1 # Number of GPUs
#SBATCH --constraint=vram48
#SBATCH -t 48:00:00 # Job time limit

source ../venv/bin/activate
python3 expt4.py > expt4.txt

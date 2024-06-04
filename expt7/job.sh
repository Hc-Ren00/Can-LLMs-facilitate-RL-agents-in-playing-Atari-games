#!/bin/bash
#SBATCH -c 2 # Number of Cores per Task
#SBATCH --mem=80G # Requested Memory
#SBATCH -p gpu-long # Partition
#SBATCH -G 1 # Number of GPUs
#SBATCH --constraint=vram48
#SBATCH -t 24:00:00 # Job time limit

source ../venv/bin/activate
python3 expt7.py > expt7.txt

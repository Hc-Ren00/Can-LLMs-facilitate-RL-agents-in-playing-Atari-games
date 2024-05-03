#!/bin/bash
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 2  # Number of GPUs
#SBATCH -t 04:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

source bin/activate
python3 main.py > 1mil.txt

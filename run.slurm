#!/bin/bash

#SBATCH -q normal
#SBATCH -p physical

#SBATCH --constraint=physg5

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-03:00:00
#SBATCH --mem=500000

module load gcccore/8.3.0
module load python/3.7.4

OMP_NUM_THREADS=1 python main.py --model ride --env procgen:procgen-coinrun-v0 --num_threads 16

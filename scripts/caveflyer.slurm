#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:30:00
#SBATCH --mem=102400

module load gcccore/8.3.0
module load python/3.7.4

OMP_NUM_THREADS=1 python ../main.py --model ride --env procgen:procgen-caveflyer-v0 --num_threads 8

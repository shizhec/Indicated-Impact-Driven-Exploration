#!/bin/bash

#SBATCH -q normal
#SBATCH -p physical

#SBATCH --constraint=physg5

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-10:00:00
#SBATCH --mem=400000

module load gcccore/8.3.0
module load foss/2019b anaconda3/2020.07 mesa/19.2.1

source /usr/local/easybuild-2019/easybuild/software/core/anaconda3/2020.07/bin/activate
conda activate clride

OMP_NUM_THREADS=1 python ../../../main.py --model vanilla --env procgen --game climber --num_threads 8 --disable_background \
--restrict_themes --use_monochrome_assets --learning_rate 0.0001 --batch_size 8 --unroll_length 100 \
--entropy_cost 0.005 --total_frames 10000000

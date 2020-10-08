#!/bin/bash

#SBATCH -q normal
#SBATCH -p physical

#SBATCH --constraint=avx512

#SBATCH --output="lvl5000_3.out"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-10:00:00
#SBATCH --mem=400000

module load gcccore/8.3.0
module load foss/2019b anaconda3/2020.07 mesa/19.2.1

source /usr/local/easybuild-2019/easybuild/software/core/anaconda3/2020.07/bin/activate
conda activate clride

OMP_NUM_THREADS=1 python ../../../main.py --model iride --env procgen --game jumper --num_threads 8 --disable_background \
--restrict_themes --use_monochrome_assets --learning_rate 0.0001 --batch_size 32 --unroll_length 40 --intrinsic_reward_coef 1.0 \
--entropy_cost 0.01 --indicator_loss_coef 10 --encoder_coef 0.5 --num_levels 5000 --start_level 200 --distribution_mode easy --total_frames 10000000

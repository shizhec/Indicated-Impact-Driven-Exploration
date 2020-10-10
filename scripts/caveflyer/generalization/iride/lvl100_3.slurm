#!/bin/bash

#SBATCH -q normal
#SBATCH -p physical

#SBATCH --constraint=avx512

#SBATCH --output="lvl100_3.out"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-05:00:00
#SBATCH --mem=250000

module load gcccore/8.3.0
module load foss/2019b anaconda3/2020.07 mesa/19.2.1

source /usr/local/easybuild-2019/easybuild/software/core/anaconda3/2020.07/bin/activate
conda activate clride

OMP_NUM_THREADS=1 python ../../../../main.py --model iride --env procgen --game caveflyer --num_threads 8 --disable_background \
--restrict_themes --use_monochrome_assets --learning_rate 0.0005 --batch_size 32 --unroll_length 60 --intrinsic_reward_coef 0.5 \
--entropy_cost 0.001 --indicator_loss_coef 1 --encoder_coef 0.25 --num_levels 100 --start_level 200 --distribution_mode easy --total_frames 10000000

#!/bin/bash

#SBATCH -q normal
#SBATCH -p physical

#SBATCH --ntasks=1
#SBATCH --time=0-12:00:00

module load gcccore/8.3.0
module load foss/2019b anaconda3/2020.07 mesa/19.2.1

source /usr/local/easybuild-2019/easybuild/software/core/anaconda3/2020.07/bin/activate
conda activate clride

python main.py --model iride --env procgen --game coinrun --modelpath lvl10000-st100 --testing --total_frames 10000000 \
--disable_background --restrict_themes --use_monochrome_assets

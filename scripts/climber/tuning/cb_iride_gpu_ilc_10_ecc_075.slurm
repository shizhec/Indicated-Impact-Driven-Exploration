#!/bin/bash

# Partition for the job:
#SBATCH --partition=gpgpu

# Multithreaded (SMP) job: must run on one node
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="climber_iride_parameter_tuning"

# The name of the output
#SBATCH --output="cb_iride_gpu_ilc_10_ecc_075.out"

# The project ID which this job should run under:
#SBATCH --account="punim1126"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1
# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-01:00:00

# memory usage
#SBATCH --mem=113000

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load gcccore/8.3.0
module load foss/2019b anaconda3/2020.07 mesa/19.2.1
module load cuda/10.1.243
module load cudnn/7.6.4.38

# activate the conda environment
source /usr/local/easybuild-2019/easybuild/software/core/anaconda3/2020.07/bin/activate
conda activate clride

# The job command(s):
OMP_NUM_THREADS=1 python ../../main.py --model iride --env procgen --game climber --num_threads 8 --disable_background \
--restrict_themes --use_monochrome_assets --learning_rate 0.0001 --batch_size 8 --unroll_length 100 --intrinsic_reward_coef 0.01 \
--entropy_cost 0.005 --indicator_loss_coef 10.0 --encoder_coef 0.75 --total_frames 4000000 --disable_checkpoint
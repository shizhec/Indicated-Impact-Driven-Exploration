#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Tue Aug 25 2020 22:30:49 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=shortgpgpu

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="gpu_test"

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

# The modules to load
module load gcccore/8.3.0
module load foss/2019b anaconda3/2020.07 mesa/19.2.1
module load cuda/10.1.243
module load cudnn/7.6.4.38

# The job command(s):
source /usr/local/easybuild-2019/easybuild/software/core/anaconda3/2020.07/bin/activate
conda activate clride

# OMP_NUM_THREADS=1 python main.py --model ride --env procgen:procgen-coinrun-v0 --num_threads 4 --num_buffers 32 --num_actors 16 --batch_size 8
# OMP_NUM_THREADS=1 python main.py --model ride --env MiniGrid-ObstructedMaze-2Dlh-v0 --num_threads 8 --use_fullobs_policy --use_fullobs_intrinsic
OMP_NUM_THREADS=1 python main.py --model ride --env procgen --game coinrun --num_threads 8 --restrict_themes --use_monochrome_assets

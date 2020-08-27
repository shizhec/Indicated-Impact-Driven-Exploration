#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Tue Aug 25 2020 22:30:49 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=gpgpu

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
#SBATCH --time=0-02:00:00

# memory usage
#SBATCH --mem=200000

# node type
#SBATCH --constraint=dlg1


# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load gcccore/8.3.0
module load python/3.7.4
module load cuda/10.2.89

# The job command(s):
OMP_NUM_THREADS=1 python main.py --model ride --env procgen:procgen-coinrun-v0 --num_threads 8


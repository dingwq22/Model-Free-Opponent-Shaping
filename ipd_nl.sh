#!/bin/bash

# Slurm sbatch options
#SBATCH --job-name ipd_nl
#SBATCH -a 0-1
## SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
#SBATCH -c 10 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022b

# Run the script

python src/main_mfos_ppo.py \
--game=IPD \
--opponent=NL \
--exp-name=runs/mfos_ppo_ipd_nl


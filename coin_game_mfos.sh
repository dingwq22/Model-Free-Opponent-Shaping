#!/bin/bash

# Slurm sbatch options
#SBATCH --job-name coin_game
#SBATCH -a 0
# SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
#SBATCH -c 10 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022b

# Run the script

python src/coin_game/main_mfos_coin_game.py \
--exp-name=runs/mfos_coin


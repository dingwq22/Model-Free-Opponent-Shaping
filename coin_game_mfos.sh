#!/bin/bash

# Slurm sbatch options
#SBATCH --job-name coin_game
#SBATCH -a 0-5
# SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
#SBATCH -c 10 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022b

# hyperparameters 
grid_size=(3 4 4 5 5 6)
num_agents=(2 2 3 3 4 4)

# Run the script
python src/coin_game/main_mfos_coin_game.py \
--exp-name=runs/coin_mfos
--coin-game-env=simple \
--grid-size=${grid_size[$SLURM_ARRAY_TASK_ID]} \
--num-agents=${num_agents[$SLURM_ARRAY_TASK_ID]}

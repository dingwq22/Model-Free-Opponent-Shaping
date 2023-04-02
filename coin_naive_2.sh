#!/bin/bash

# Slurm sbatch options
#SBATCH --job-name coin_naive2
#SBATCH -a 0-3
# SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
#SBATCH -c 10 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022b
source ~/proxy.env

logs_folder="out_coin_naive2"
mkdir -p $logs_folder

# hyperparameters 
seed_max=1
grid_size=(3 4 5 10)
num_agents=2
num_coins=2

# Run the script
for seed in `seq ${seed_max}`;
do
echo "seed: ${seed}"
python src/coin_game/main_naive_coin_game.py \
--project_name "coin_naive_${num_agents}" \
--exp_name "naive_${grid_size[$SLURM_ARRAY_TASK_ID]}_${num_coins[$SLURM_ARRAY_TASK_ID]}" \
--seed ${seed} \
--env_name "simple" \
--grid_size ${grid_size[$SLURM_ARRAY_TASK_ID]} \
--num_agents ${num_agents} \
--num_coins ${num_coins} \
--user_name "mfos" \
&> $logs_folder/out_${grid_size[$SLURM_ARRAY_TASK_ID]_num_agents[$SLURM_ARRAY_TASK_ID]}_${num_coins[$SLURM_ARRAY_TASK_ID]}_${seed}
done
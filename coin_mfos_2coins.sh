#!/bin/bash

# Slurm sbatch options
#SBATCH --job-name coin_mfos_2coins
#SBATCH -a 0-9
# SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
#SBATCH -c 10 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022b
source ~/proxy.env

logs_folder="out_coin_mfos_2coins"
mkdir -p $logs_folder

# hyperparameters 
seed_max=1
grid_size=(3 3 4 4 5 5 5 9 9 9)
num_agents=(1 2 1 2 1 2 3 1 2 4)
num_coins=(2 2 2 2 2 2 2 2 2 2)

# Run the script
for seed in `seq ${seed_max}`;
do
echo "seed: ${seed}"
python src/coin_game/main_mfos_coin_game.py \
--project_name "coin_mfos_2coin" \
--exp_name "mfos_${grid_size[$SLURM_ARRAY_TASK_ID]}_${num_agents[$SLURM_ARRAY_TASK_ID]}_${num_coins[$SLURM_ARRAY_TASK_ID]}" \
--seed ${seed} \
--env_name "multi" \
--grid_size ${grid_size[$SLURM_ARRAY_TASK_ID]} \
--num_agents ${num_agents[$SLURM_ARRAY_TASK_ID]} \
--num_coins ${num_coins[$SLURM_ARRAY_TASK_ID]} \
--user_name "mfos" \
&> $logs_folder/out_2coin_${grid_size[$SLURM_ARRAY_TASK_ID]}_${num_agents[$SLURM_ARRAY_TASK_ID]}_${num_coins[$SLURM_ARRAY_TASK_ID]}_${seed}
done

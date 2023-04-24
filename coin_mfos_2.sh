#!/bin/bash

# Slurm sbatch options
#SBATCH --job-name coin_mfos2
#SBATCH -a 0-3
# SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
#SBATCH -c 10 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022b
source ~/proxy.env

logs_folder="out_coin_mfos2"
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
python src/coin_game/main_mfos_coin_game.py \
--project_name "coin_mfos_${num_agents}" \
--exp_name "mfos_${grid_size[$SLURM_ARRAY_TASK_ID]}_${num_agents}_${num_coins}" \
--seed ${seed} \
--env_name "multi" \
--grid_size ${grid_size[$SLURM_ARRAY_TASK_ID]} \
--num_agents ${num_agents} \
--num_coins ${num_coins} \
--user_name "mfos" \
&> $logs_folder/out_${grid_size[$SLURM_ARRAY_TASK_ID]}_${num_agents}_${num_coins}_${seed}
done


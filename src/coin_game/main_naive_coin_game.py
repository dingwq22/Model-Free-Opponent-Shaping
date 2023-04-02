#!/usr/bin/env python
import torch
import json
from coin_game_envs import CoinGameGPU, CoinGameGPU_MultiPlayer
from coin_game_ppo_agent import PPO, Memory
import argparse
import wandb
import socket 
import setproctitle
import numpy as np
from pathlib import Path

import os,sys
sys.path.append(os.path.abspath(os.getcwd()))
from utils.utils import print_args, print_box, connected_to_internet


parser = argparse.ArgumentParser()
parser.add_argument("--project_name", type=str, default="test")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--exp_name", type=str, default="")
parser.add_argument("--env_name", type=str, default="simple")
parser.add_argument("--grid_size", type=int, default=3)
parser.add_argument("--num_agents", type=int, default=2,
                    help="number of agents for each color")
parser.add_argument("--num_coins", type=int, default=2,
                    help="number of coins for each color") 
parser.add_argument("--user_name", type=str, default='mfos',
                    help="[for wandb usage], to specify user's name for "
                        "simply collecting training data.")
parser.add_argument("--use_wandb", action='store_false', default=True, 
                    help="[for wandb usage], by default True, will log date "
                        "to wandb server. or else will use tensorboard to log data.")
args = parser.parse_args()

if __name__ == "__main__":
    ############## Hyperparameters ##############
    batch_size = 512  # 8192 #, 32768
    state_dim = [4, args.grid_size, args.grid_size]
    action_dim = 4
    n_latent_var = 8  # number of variables in hidden layer

    lr = 0.005
    betas = (0.9, 0.999)
    gamma = 0.96  # discount factor

    traj_length = 32

    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    use_gae = False

    inner_ep_len = 32
    num_steps = 512  # , 500
    max_episodes = num_steps // inner_ep_len  # max training episodes
    name = args.exp_name

    # print(f"RUNNING NAME: {name}")
    # if not os.path.isdir(name):
    #     os.mkdir(name)
    #     with open(os.path.join(name, "commandline_args.txt"), "w") as f:
    #         json.dump(args.__dict__, f, indent=2)

    ##################
    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + 
        "/results") / args.env_name / args.project_name / args.exp_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if args.use_wandb:

        # for supercloud when no internet_connection
        if not connected_to_internet():
            import json
            # save a json file with your wandb api key in your 
            # home folder as {'my_wandb_api_key': 'INSERT API HERE'}
            # NOTE this is only for running on systems without internet access
            # have to run `wandb sync wandb/run_name` to sync logs to wandboard
            with open(os.path.expanduser('~')+'/keys.json') as json_file: 
                key = json.load(json_file)
                my_wandb_api_key = key['my_wandb_api_key'] # NOTE change here as well
            os.environ["WANDB_API_KEY"] = my_wandb_api_key
            os.environ["WANDB_MODE"] = "dryrun"
            os.environ['WANDB_SAVE_CODE'] = "true"

        print_box('Creating wandboard...')
        run = wandb.init(config=args,
                        project=args.project_name,
                        entity=args.user_name,
                        notes=socket.gethostname(),
                        name=str(args.env_name) + "_" +
                        str(args.exp_name) +
                        "_seed" + str(args.seed),
                        # group=all_args.scenario_name,
                        dir=str(run_dir),
                        # job_type="training",
                        reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for 
                            folder in run_dir.iterdir() if 
                            str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(args.env_name) + "-" + \
                            str(args.exp_name) + "@" + \
                            str(args.user_name))

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    config = {
        "all_args": args,
        "num_agents": args.num_agents,
        "run_dir": run_dir
    }


    #############################################

    memory_0 = Memory()
    ppo_0 = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    memory_1 = Memory()
    ppo_1 = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    print(lr, betas)
    print(sum(p.numel() for p in ppo_0.policy_old.parameters() if p.requires_grad))
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    rew_means = []

    # env
    if args.env_name == "multi":
        env = CoinGameGPU_MultiPlayer(batch_size=batch_size, 
                                max_steps=inner_ep_len - 1, 
                                grid_size=args.grid_size,
                                num_agents=args.num_agents,
                                num_coins=args.num_coins)
    else:
        env = CoinGameGPU(batch_size=batch_size, 
                          max_steps=inner_ep_len - 1, 
                          grid_size=args.grid_size)

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        running_reward = 0
        opp_running_reward = 0
        p1_num_opp, p2_num_opp, p1_num_self, p2_num_self = 0, 0, 0, 0
        for t in range(inner_ep_len):
            # Running policy_old:
            with torch.no_grad():
                action_0 = ppo_0.policy_old.act(state[0].detach(), memory_0)
                action_1 = ppo_1.policy_old.act(state[1].detach(), memory_1)

            state, reward, done, info_2 = env.step([action_0.detach(), action_1.detach()])

            running_reward += reward[0].detach()
            opp_running_reward += reward[1].detach()
            memory_0.rewards.append(reward[0].detach())
            memory_1.rewards.append(reward[1].detach())
            if info_2 is not None:
                p1_num_opp += info_2[2]
                p2_num_opp += info_2[1]
                p1_num_self += info_2[3]
                p2_num_self += info_2[0]

        ppo_0.update(memory_0)
        ppo_1.update(memory_1)
        memory_0.clear_memory()
        memory_1.clear_memory()

        rew_means.append(
            {
                "episode": i_episode,
                "rew": running_reward.mean().item(),
                "opp_rew": opp_running_reward.mean().item(),
                "p1_opp": p1_num_opp.float().mean().item(),
                "p2_opp": p2_num_opp.float().mean().item(),
                "p1_self": p1_num_self.float().mean().item(),
                "p2_self": p2_num_self.float().mean().item(),
            }
        )
        print(rew_means[-1])

    # ppo_0.save(os.path.join(name, f"{i_episode}_0.pth"))
    # ppo_1.save(os.path.join(name, f"{i_episode}_1.pth"))
    # with open(os.path.join(name, f"out_{args.env_name}_{args.grid_size}_{args.num_agents}.json"), "w") as f:
    #     json.dump(rew_means, f)
    # print(f"SAVING! {i_episode}")

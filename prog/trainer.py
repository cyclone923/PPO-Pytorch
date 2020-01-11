import gym
import torch

import sys
sys.path.append("..")
import algorithm
from tool.memory import Memory
from tool.settings import get_env_setting


class Trainer:
    def __init__(self, args):
        self.env_name = args.environment
        self.alg_name = args.algorithm
        self.net_name = args.network

        self.env_setting = get_env_setting(self.env_name)
        self.update_timestep = self.env_setting["update_timestep"]
        self.max_timesteps = self.env_setting["max_timesteps"]
        self.solved_reward = self.env_setting["solved_reward"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(args.environment)
        self.alg = self.pick_alg(args, self.env, self.device)
        print("Algorithm Used: {}".format(args.algorithm))
        self.log_interval = 20 # print avg reward in the interval
        self.max_episodes = 100000
        self.render = False

        if args.seed:
            torch.manual_seed(args.seed)
            self.env.seed(args.seed)

    def pick_alg(self, args, env, device):
        if args.algorithm == "a2c":
            alg = algorithm.A2C(args, env, device)
        elif args.algorithm == "ppo":
            alg = algorithm.PPO(args, env, device, self.env_setting["ppo_k_updates"])
        else:
            raise NotImplementedError("Algorithm not implemented")
        return alg

    def train(self):
        # logging variables
        running_reward = 0
        avg_length = 0
        time_step = 0

        memory = Memory()

        self.alg.load_dict("./", self.env_name, self.alg_name, self.net_name)

        # training loop
        for i_episode in range(1, self.max_episodes + 1):
            obs = self.env.reset()
            self.alg.memory_reset()
            for t in range(self.max_timesteps):
                if t == self.max_timesteps:
                    print(f"Reach maximum step {t}")
                    exit(0)
                time_step += 1

                # Running policy_old:
                action = self.alg.take_action(obs, memory)
                obs, reward, done, _ = self.env.step(action)

                # Saving reward and is_terminal:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                # update if its time
                if time_step >= self.update_timestep and done == True:
                    self.alg.update(memory)
                    memory.clear_memory()
                    time_step = 0

                running_reward += reward
                if self.render:
                    self.env.render()
                if done:
                    break

            avg_length += t

            # stop training if avg_reward > solved_reward or reaches the limit of training epoches
            if running_reward > (self.log_interval * self.solved_reward):
                print("########## Solved! ##########")
                directory = "./preTrained/"
                self.alg.save_dict(directory, self.env_name, self.alg_name, self.net_name)
                break

            # save every 500 episodes
            if i_episode % 500 == 0:
                directory = "./"
                self.alg.save_dict(directory, self.env_name, self.alg_name, self.net_name)

            # logging
            if i_episode % self.log_interval == 0:
                avg_length = int(avg_length / self.log_interval)
                running_reward = int((running_reward / self.log_interval))

                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
                running_reward = 0
                avg_length = 0


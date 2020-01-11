import gym
import torch
from PIL import Image

import sys
sys.path.append("..")
import algorithm
from tool.memory import Memory
from tool.settings import get_env_setting


class Tester:
    def __init__(self, args):
        self.env_name = args.environment
        self.alg_name = args.algorithm
        self.net_name = args.network

        self.env_setting = get_env_setting(self.env_name)
        self.max_timesteps = self.env_setting["max_timesteps"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(args.environment)
        self.alg = self.pick_alg(args, self.env, self.device)
        print("Algorithm Used: {}".format(args.algorithm))

        self.render = True
        self.save_gif = False

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

    def test(self, n_episodes):
        memory = Memory()

        self.alg.load_dict("./", self.alg_name, self.env_name, self.net_name)

        # testing loop
        for ep in range(1, n_episodes + 1):
            ep_reward = 0
            obs = self.env.reset()
            for t in range(self.max_timesteps):
                if t == self.max_timesteps:
                    print(f"Reach maximum step {t}")
                    exit(0)
                action = self.alg.take_action(obs, memory)
                obs, reward, done, _ = self.env.step(action)
                ep_reward += reward
                if self.render:
                    self.env.render()
                if self.save_gif:
                    img = self.env.render(mode='rgb_array')
                    img = Image.fromarray(img)
                    img.save('./gif/{}.jpg'.format(t))
                if done:
                    break

            print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
            self.env.close()


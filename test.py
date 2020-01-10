import gym
from PIL import Image
import torch
import algorithm
from tool.memory import Memory
from argparse import ArgumentParser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pick_alg(args, env):
    if args.algorithm == "a2c":
        alg = algorithm.A2C(args, env, device)
    elif args.algorithm == "ppo":
        alg = algorithm.PPO(args, env, device)
    else:
        raise NotImplementedError("Algorithm not implemented")
    return alg

def parse_args():
    parser = ArgumentParser()

    # crucial arguments
    parser.add_argument('-s', '--seed', default=None, type=int,
                        help='random seed for torch and gym')
    parser.add_argument('-l', '--lr', default=0.002, type=float,
                        help='learning rate')
    parser.add_argument('-b', '--betas', default=(0.9, 0.99), type=tuple,
                        help='hyper-parameter for Adam optimizer')
    parser.add_argument('-g', '--gamma', default=0.99, type=float,
                        help='discount factor for future reward')

    parser.add_argument('-a', '--algorithm', default="ppo", type=str,
                        help='algorithm use for training the agent')
    parser.add_argument('-e', '--environment', default="LunarLander-v2", type=str,
                        help='environment used for training')
    parser.add_argument('-n', '--network', default="rc", type=str,
                        help='network used for function approximation')

    # optional arguments
    parser.add_argument('-k', '--k_epochs', default=4, type=int,
                        help='update old parameters every k updates for ppo')
    parser.add_argument('-c', '--eps_clip', default=0.2, type=float,
                        help='epsilon clip co-efficient for ppo')
    parser.add_argument('-d', '--action_std', default=None, type=float,
                        help='constant standard deviation to sample an action from a diagonal multivariate normal')

    args = parser.parse_args()
    return args


def main():
    n_episodes = 30
    max_timesteps = 3000
    render = True
    save_gif = False

    args = parse_args()
    env = gym.make(args.environment)
    alg = pick_alg(args, env)
    memory = Memory()
    print("Algorithm Used: {}".format(args.algorithm))

    filename = "{}_{}_{}.pth".format(args.algorithm, args.environment, args.network)
    directory = "./preTrained/"
    alg.act_policy().load_state_dict(torch.load(directory+filename))
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        obs = env.reset()
        for t in range(max_timesteps):
            action = alg.take_action(obs, memory)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))  
            if done:
                break
            
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        env.close()
    
if __name__ == '__main__':
    main()
    
    

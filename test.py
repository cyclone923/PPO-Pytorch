import gym
from PIL import Image
import torch
import algorithm
from tool.memory import Memory
from argparse import ArgumentParser

def pick_alg(name, env, args):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    if name == "a2c":
        alg = algorithm.A2C(state_dim, action_dim, args.n_latent_var, args.lr, args.betas, args.gamma)
    elif name == "ppo":
        alg = algorithm.PPO(state_dim, action_dim, args.n_latent_var, args.lr, args.betas, args.gamma, args.k_epochs, args.eps_clip)
    else:
        raise NotImplementedError("Algorithm not implemented")
    return alg

def parse_args():
    parser = ArgumentParser()

    # crucial arguments
    parser.add_argument('-n', '--n_latent_var', default=32, type=int,
                        help='number of nodes in the hidden layer of neural network')
    parser.add_argument('-s', '--seed', default=22, type=int,
                        help='random seed for torch and gym')
    parser.add_argument('-l', '--lr', default=0.002, type=float,
                        help='learning rate')
    parser.add_argument('-b', '--betas', default=(0.9, 0.99), type=tuple,
                        help='hyper-parameter for Adam optimizer')
    parser.add_argument('-g', '--gamma', default=0.99, type=float,
                        help='discount factor for future reward')

    # optional arguments
    parser.add_argument('-k', '--k_epochs', default=4, type=int,
                        help='update old parameters every k updates for ppo')
    parser.add_argument('-e', '--eps_clip', default=0.2, type=float,
                        help='epsilon clip co-efficient for ppo')

    args = parser.parse_args()
    return args

def test():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v2"
    alg_name = "a2c"
    #############################################

    n_episodes = 3
    max_timesteps = 3000
    render = True
    save_gif = False

    filename = "{}_{}.pth".format(alg_name, env_name)
    directory = "./preTrained/"

    args = parse_args()
    env = gym.make(env_name)
    alg = pick_alg(alg_name, env, args)
    memory = Memory()
    print("Algorithm Used: {}".format(alg_name))
    alg.act_policy().load_state_dict(torch.load(directory+filename))
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = alg.take_action(state, memory)
            state, reward, done, _ = env.step(action)
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
    test()
    
    

import torch
import gym
import os
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
    parser.add_argument('-s', '--seed', default=None, type=int,
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

def main():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v2"
    alg_name = "a2c"
    render = False

    solved_reward = 230               # stop training if avg_reward > solved_reward
    log_interval = 20                 # print avg reward in the interval
    max_episodes = 20000              # max training episodes
    max_timesteps = 300               # max timesteps in one episode
    update_timestep = 2000            # update policy every n timesteps
    #############################################

    args = parse_args()
    env = gym.make(env_name)
    alg = pick_alg(alg_name, env, args)
    memory = Memory()
    print("Algorithm Used: {}".format(alg_name))

    if args.seed:
        torch.manual_seed(args.seed)
        env.seed(args.seed)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1
            
            # Running policy_old:
            action = alg.act(state, memory)
            state, reward, done, _ = env.step(action)
            
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            if timestep >= update_timestep and done == True:
                alg.update(memory)
                memory.clear_memory()
                timestep = 0
            
            running_reward += reward
            if render:
                env.render()
            if done:
                break
                
        avg_length += t
        
        # stop training if avg_reward > solved_reward or reaches the limit of training epoches
        if running_reward > (log_interval*solved_reward) or i_episode == max_episodes:
            print("########## Solved! ##########")
            directory = "./preTrained/"
            torch.save(alg.policy_dict(), os.path.join(directory, '{}_{}.pth'.format(alg_name, env_name)))
            break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

            
if __name__ == '__main__':
    main()

    

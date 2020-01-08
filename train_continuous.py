import torch
import gym
import os
import algorithm
from tool.memory import Memory
from argparse import ArgumentParser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pick_alg(name, env, args):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    if name == "a2c":
        alg = algorithm.A2C(obs_dim, action_dim, args.n_latent_var, args.lr, args.betas, args.gamma, device, args.action_std)
    elif name == "ppo":
        alg = algorithm.PPO(obs_dim, action_dim, args.n_latent_var, args.lr, args.betas, args.gamma, args.k_epochs, args.eps_clip, device, args.action_std)
    else:
        raise NotImplementedError("Algorithm not implemented")
    return alg

def parse_args():
    parser = ArgumentParser()

    # crucial arguments
    parser.add_argument('-n', '--n_latent_var', default=64, type=int,
                        help='number of nodes in the hidden layer of neural network')
    parser.add_argument('-s', '--seed', default=None, type=int,
                        help='random seed for torch and gym')
    parser.add_argument('-l', '--lr', default=0.0001, type=float,
                        help='learning rate')
    parser.add_argument('-b', '--betas', default=(0.9, 0.99), type=tuple,
                        help='hyper-parameter for Adam optimizer')
    parser.add_argument('-g', '--gamma', default=0.99, type=float,
                        help='discount factor for future reward')
    parser.add_argument('-a', '--action_std', default=0.5, type=float,
                        help='constant standard deviation to sample an action from a diagonal multivariate normal')

    # optional arguments
    parser.add_argument('-k', '--k_epochs', default=80, type=int,
                        help='update old parameters every k updates for ppo')
    parser.add_argument('-e', '--eps_clip', default=0.2, type=float,
                        help='epsilon clip co-efficient for ppo')

    args = parser.parse_args()
    return args

def main():
    ############## Hyperparameters ##############
    env_name = "BipedalWalker-v2"
    alg_name = "ppo"
    render = False
    solved_reward = 260         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 20000        # max training episodes
    max_timesteps = 3000        # max timesteps in one episode

    update_timestep = 4000      # update policy every n timesteps

    #############################################

    # creating environment
    args = parse_args()
    env = gym.make(env_name)
    alg = pick_alg(alg_name, env, args)
    memory = Memory()
    print("Algorithm Used: {}".format(alg_name))

    filename = "{}_{}.pth".format(alg_name, env_name)
    directory = "./preTrained/"
    alg.act_policy().load_state_dict(torch.load(directory+filename))

    if args.seed:
        print("Random Seed: {}".format(args.seed))
        torch.manual_seed(args.seed)
        env.seed(args.seed)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        obs = env.reset()
        for t in range(max_timesteps):
            if t == max_timesteps:
                print(f"Reach maximum step {t}")
                exit(0)
            time_step +=1
            # Running policy_old:
            action = alg.take_action(obs, memory)
            obs, reward, done, _ = env.step(action)

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step >= update_timestep and done == True:
                alg.update(memory)
                memory.clear_memory()
                time_step = 0
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

        # # save every 500 episodes
        # if i_episode % 500 == 0:
        #     torch.save(alg.policy_dict(), './PPO_continuous_{}.pth'.format(env_name))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()




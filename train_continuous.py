import torch
import gym
import os
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
    parser.add_argument('-l', '--lr', default=0.0001, type=float,
                        help='learning rate')
    parser.add_argument('-b', '--betas', default=(0.9, 0.99), type=tuple,
                        help='hyper-parameter for Adam optimizer')
    parser.add_argument('-g', '--gamma', default=0.99, type=float,
                        help='discount factor for future reward')

    parser.add_argument('-a', '--algorithm', default="ppo", type=str,
                        help='algorithm use for training the agent')
    parser.add_argument('-e', '--environment', default="BipedalWalker-v2", type=str,
                        help='environment used for training')
    parser.add_argument('-n', '--network', default="rc", type=str,
                        help='network used for function approximation')

    # optional arguments
    parser.add_argument('-k', '--k_epochs', default=20, type=int,
                        help='update old parameters every k updates for ppo')
    parser.add_argument('-c', '--eps_clip', default=0.2, type=float,
                        help='epsilon clip co-efficient for ppo')
    parser.add_argument('-d', '--action_std', default=0.3, type=float,
                        help='constant standard deviation to sample an action from a diagonal multivariate normal')

    args = parser.parse_args()
    return args


def main():
    ############## Hyperparameters ##############
    render = False
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 100000       # max training episodes
    max_timesteps = 3000        # max timesteps in one episode
    update_timestep = 4000      # update policy every n timesteps
    #############################################

    args = parse_args()
    env = gym.make(args.environment)
    alg = pick_alg(args, env)
    memory = Memory()
    print("Algorithm Used: {}".format(args.algorithm))

    filename = "{}_{}_{}.pth".format(args.algorithm, args.environment, args.network)
    alg.policy_old.load_state_dict(torch.load(filename))
    alg.policy.load_state_dict(torch.load(filename))

    if args.seed:
        torch.manual_seed(args.seed)
        env.seed(args.seed)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        obs = env.reset()
        alg.memory_reset()
        for t in range(max_timesteps):
            if t == max_timesteps:
                print(f"Reach maximum step {t}")
                exit(0)
            time_step += 1

            # Running policy_old:
            action = alg.take_action(obs, memory)
            obs, reward, done, _ = env.step(action)

            # Saving reward and is_terminal:
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
        if running_reward > (log_interval * solved_reward) or i_episode == max_episodes:
            print("########## Solved! ##########")
            directory = "./preTrained/"
            torch.save(alg.policy_dict(), os.path.join(directory, '{}_{}_{}.pth'.format(args.algorithm, args.environment, args.network)))
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(alg.policy_dict(), '{}_{}_{}.pth'.format(args.algorithm, args.environment, args.network))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()





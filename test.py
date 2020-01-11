from argparse import ArgumentParser
from prog.tester import Tester

def parse_args():
    parser = ArgumentParser()

    # crucial arguments
    parser.add_argument('-s', '--seed', default=None, type=int,
                        help='random seed for torch and gym')
    parser.add_argument('-l', '--lr', default=0.001, type=float,
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
    parser.add_argument('-c', '--eps_clip', default=0.2, type=float,
                        help='epsilon clip co-efficient for ppo')
    parser.add_argument('-d', '--action_std', default=None, type=float,
                        help='constant standard deviation to sample an action from a diagonal multivariate normal')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    prog = Tester(args)
    prog.test(5)

if __name__ == '__main__':
    main()


    
    

import torch
import torch.nn as nn
from .network.fc_ac import FC_ActorCritic
from .network.rc_ac import RC_ActorCritic

net_work_selection = {"fc": FC_ActorCritic, "rc": RC_ActorCritic}

class PPO:
    def __init__(self, args, env, device):

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.lr = args.lr
        self.betas = args.betas
        self.gamma = args.gamma
        self.eps_clip = args.eps_clip
        self.k_epochs = args.k_epochs
        self.device = device

        self.net_work = net_work_selection[args.network]
        self.policy = self.net_work(obs_dim, action_dim, device, args.action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.policy_old = self.net_work(obs_dim, action_dim, device, args.action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def policy_dict(self):
        return self.policy.state_dict()

    def act_policy(self):
        return self.policy_old

    def memory_reset(self):
        self.policy.reset()

    def take_action(self, state, memory):
        return self.act_policy().act(state, memory)

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.append(discounted_reward)
        rewards = list(reversed(rewards))

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(memory)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
import torch
import torch.nn as nn
from .network.fc_ac import FC_ActorCritic



class A2C:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, device, action_std=None):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.device = device

        self.policy = FC_ActorCritic(state_dim, action_dim, n_latent_var, device, action_std).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.MseLoss = nn.MSELoss()

    def policy_dict(self):
        return self.policy.state_dict()

    def act_policy(self):
        return self.policy

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
        states = torch.stack(memory.states).to(self.device).detach()
        actions = torch.stack(memory.actions).to(self.device).detach()

        # Evaluating old actions and values :
        logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)

        # Finding Surrogate Loss:
        advantages = rewards - state_values.detach()
        loss = -logprobs * advantages + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


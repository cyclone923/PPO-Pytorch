import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, action_std=None):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var // 2),
            nn.Tanh(),
            nn.Linear(n_latent_var // 2, action_dim),
        )
        if action_std is None:
            self.action_layer.add_module("soft_max", nn.Softmax(dim=-1))
            self.action_distribution = Categorical
        else:
            self.action_distribution = MultivariateNormal
            self.cov_mat = torch.diag(action_std * action_std * torch.ones(action_dim)).to(device)

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var // 2),
            nn.Tanh(),
            nn.Linear(n_latent_var // 2, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.FloatTensor(state).to(device)
        action_mean = self.action_layer(state)
        if hasattr(self, "cov_mat"):
            dist = self.action_distribution(action_mean, self.cov_mat)
        else:
            dist = self.action_distribution(action_mean)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.cpu().numpy()

    def evaluate(self, state, action):
        action_mean = self.action_layer(state)
        if hasattr(self, "cov_mat"):
            dist = self.action_distribution(action_mean, self.cov_mat.repeat(action_mean.size()[0], 1, 1).to(device))
        else:
            dist = self.action_distribution(action_mean)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class A2C:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, action_std=None):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)
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
        rewards = torch.tensor(rewards).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        states = torch.stack(memory.states).to(device).detach()
        actions = torch.stack(memory.actions).to(device).detach()

        # Evaluating old actions and values :
        logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)

        # Finding Surrogate Loss:
        advantages = rewards - state_values.detach()
        loss = -logprobs * advantages + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


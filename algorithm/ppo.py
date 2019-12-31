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


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, action_std=None):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def policy_dict(self):
        return self.policy.state_dict()

    def act_policy(self):
        return self.policy_old

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
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

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
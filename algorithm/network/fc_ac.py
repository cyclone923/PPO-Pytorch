import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal


class FC_ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, device, action_std=None):
        super(FC_ActorCritic, self).__init__()

        self.device = device

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
            self.cov_mat = torch.diag(action_std * action_std * torch.ones(action_dim))

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
        state = torch.FloatTensor(state).to(self.device)
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
            dist = self.action_distribution(action_mean, self.cov_mat.repeat(action_mean.size()[0], 1, 1).to(self.device))
        else:
            dist = self.action_distribution(action_mean)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy
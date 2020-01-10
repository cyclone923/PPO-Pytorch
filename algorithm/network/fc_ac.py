import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal

class FC_ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device, action_std=None):
        super(FC_ActorCritic, self).__init__()

        self.device = device
        self.n_latent_var = 32

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, self.n_latent_var),
            nn.Tanh(),
            nn.Linear(self.n_latent_var, self.n_latent_var // 2),
            nn.Tanh(),
            nn.Linear(self.n_latent_var // 2, action_dim),
        )
        if action_std is None:
            self.action_layer.add_module("soft_max", nn.Softmax(dim=-1))
            self.action_distribution = Categorical
        else:
            self.action_distribution = MultivariateNormal
            self.cov_mat = torch.diag(action_std * action_std * torch.ones(action_dim))

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, self.n_latent_var),
            nn.Tanh(),
            nn.Linear(self.n_latent_var, self.n_latent_var // 2),
            nn.Tanh(),
            nn.Linear(self.n_latent_var // 2, 1)
        )

    def forward(self):
        raise NotImplementedError

    def reset(self):
        pass

    def act(self, obs, memory):
        obs = torch.FloatTensor(obs).to(self.device)
        action_mean = self.action_layer(obs)
        if hasattr(self, "cov_mat"):
            dist = self.action_distribution(action_mean, self.cov_mat)
        else:
            dist = self.action_distribution(action_mean)
        action = dist.sample()

        memory.obs.append(obs)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.cpu().numpy()

    def evaluate(self, memory):

        obs = torch.stack(memory.obs).to(self.device).detach()
        actions = torch.stack(memory.actions).to(self.device).detach()

        action_mean = self.action_layer(obs)
        if hasattr(self, "cov_mat"):
            dist = self.action_distribution(action_mean, self.cov_mat.repeat(action_mean.size()[0], 1, 1).to(self.device))
        else:
            dist = self.action_distribution(action_mean)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(obs)

        return action_logprobs, torch.squeeze(state_value), dist_entropy
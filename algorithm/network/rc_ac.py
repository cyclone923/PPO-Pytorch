import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal

class RC_ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device, action_std=None):
        super(RC_ActorCritic, self).__init__()

        self.device = device

        # actor
        self.n_gru_hidden = 32
        self.n_action_hidden = 32
        self.gru_h = torch.zeros( size=(1, self.n_gru_hidden) )
        self.gru_cell = nn.GRUCell(input_size=state_dim, hidden_size=self.n_gru_hidden)
        self.action_layer = nn.Sequential(
            nn.Linear(self.n_gru_hidden, self.n_action_hidden),
            nn.Tanh(),
            nn.Linear(self.n_action_hidden, self.n_action_hidden // 2),
            nn.Tanh(),
            nn.Linear(self.n_action_hidden // 2, action_dim),
        )

        if action_std is None:
            self.action_layer.add_module("soft_max", nn.Softmax(dim=-1))
            self.action_distribution = Categorical
        else:
            self.action_distribution = MultivariateNormal
            self.cov_mat = torch.diag(action_std * action_std * torch.ones(action_dim))

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(self.n_gru_hidden, self.n_action_hidden),
            nn.Tanh(),
            nn.Linear(self.n_action_hidden, self.n_action_hidden // 2),
            nn.Tanh(),
            nn.Linear(self.n_action_hidden // 2, 1)
        )

    def forward(self):
        raise NotImplementedError

    def reset(self):
        self.gru_h = torch.zeros( size=(1, self.n_gru_hidden) )

    def act(self, obs, memory):
        obs = torch.FloatTensor(obs).to(self.device)

        memory.obs.append(obs)
        memory.h_states.append(self.gru_h[0])
        self.gru_h = self.gru_cell(obs.expand(1, -1), self.gru_h)

        action_mean = self.action_layer(self.gru_h[0])
        if hasattr(self, "cov_mat"):
            dist = self.action_distribution(action_mean, self.cov_mat)
        else:
            dist = self.action_distribution(action_mean)
        action = dist.sample()


        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.cpu().numpy()

    def evaluate(self, memory):

        obs = torch.stack(memory.obs).to(self.device).detach()
        actions = torch.stack(memory.actions).to(self.device).detach()
        h_states = torch.stack(memory.h_states).to(self.device).detach()

        h_states_new = self.gru_cell(obs, h_states)
        action_mean = self.action_layer(h_states_new)

        if hasattr(self, "cov_mat"):
            dist = self.action_distribution(action_mean, self.cov_mat.repeat(action_mean.size()[0], 1, 1).to(self.device))
        else:
            dist = self.action_distribution(action_mean)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(h_states_new)

        return action_logprobs, torch.squeeze(state_value), dist_entropy
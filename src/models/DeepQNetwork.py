import gin
import torch
import torch.nn as nn

from utils.utils import create_fully_connected_network


@gin.configurable
class DeepQNetwork(nn.Module):
    def __init__(self, observation_shape, action_space, fc_sizes):
        super().__init__()
        self.input_dim = observation_shape[0]
        self.action_space = action_space
        self.model = create_fully_connected_network([self.input_dim, *fc_sizes, self.action_space.n])

    def forward(self, observations):
        return self.model(observations)

    def get_q_values(self, observations, actions):
        q_values = self(observations)
        return q_values.gather(1, actions.unsqueeze(1)).squeeze()

    @torch.no_grad()
    def get_max_q_values(self, observations):
        q_values = self(observations)
        max_q_values, _ = torch.max(q_values, dim=1)
        return max_q_values

    @torch.no_grad()
    def get_max_actions(self, observations):
        q_values = self(observations)
        _, max_action = torch.max(q_values, dim=1)
        return max_action

    @torch.no_grad()
    def sample(self, observation, epsilon):
        if torch.rand(1).item() < epsilon:
            return self.action_space.sample()

        q_values = self(observation)
        return torch.argmax(q_values).item()

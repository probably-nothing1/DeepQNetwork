# import gin
import numpy as np
import torch
import torch.nn as nn


class DeepQNetwork(nn.Module):
    def __init__(self, observation_shape, action_space):
        super().__init__()
        self.input_channels = observation_shape[0]
        self.action_space = action_space
        self.conv_net = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        flatten_size = self._compute_conv_out(observation_shape)
        self.fc_net = nn.Sequential(nn.Linear(flatten_size, 512), nn.ReLU(), nn.Linear(512, self.action_space.n))

    def _compute_conv_out(self, observation_shape):
        dummy_input = torch.zeros(1, *observation_shape)
        with torch.no_grad():
            out = self.conv_net(dummy_input)
        return int(np.prod(out.shape))

    def forward(self, observations):
        if len(observations.shape) == 3:
            observations.unsqueeze_(0)
        conv_out = self.conv_net(observations).view(observations.size(0), -1)
        q_values = self.fc_net(conv_out)
        return q_values

    def get_q_values(self, observations, actions):
        q_values = self(observations)
        return q_values.gather(0, actions.unsqueeze(1)).squeeze()

    @torch.no_grad()
    def get_max_q_values(self, observations):
        q_values = self(observations)
        max_q_values, _ = torch.max(q_values, dim=1)
        return max_q_values

    @torch.no_grad()
    def sample(self, observations, epsilon):
        if torch.rand(1).item() < epsilon:
            return self.action_space.sample()

        q_values = self(observations)
        return torch.argmax(q_values, dim=1).item()

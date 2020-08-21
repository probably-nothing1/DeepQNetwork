# import gin
import numpy as np
import torch
import torch.nn as nn

# from utils.utils import create_fully_connected_network

# def train_actor(actor, data, optimizer):
#     optimizer.zero_grad()
#     observations = data["observations"]
#     actions = data["actions"]
#     theta = data["theta"]

#     _, policy = actor(observations)
#     log_probs = policy.log_prob(actions)
#     loss = -(theta * log_probs).mean()

#     entropy = policy.entropy().mean()

#     loss.backward()
#     optimizer.step()
#     return loss.item(), entropy.item()


# @gin.configurable
# def create_deep_q_network(observation_space, action_space, hidden_sizes):
#     observation_dim = observation_space.shape[0]
#     action_dim = action_space.n
#     return DeepQNetwork(observation_dim, action_dim, hidden_sizes)


class DeepQNetwork(nn.Module):
    def __init__(self, observation_shape, num_of_actions):
        super().__init__()
        self.input_channels = observation_shape[0]
        self.conv_net = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3),
            nn.ReLU(),
        )
        flatten_size = self._compute_conv_out(observation_shape)
        self.fc_net = nn.Sequential(
            nn.Linear(flatten_size, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_of_actions),
        )

    def _compute_conv_out(self, observation_shape):
        dummy_input = torch.zeros(1, *observation_shape)
        with torch.no_grad():
            out = self.conv_net(dummy_input)
        return int(np.prod(out.shape))

    def forward(self, observation):
        batch_size = observation.size()[0]
        conv_out = self.conv_net(observation).view(batch_size, -1)
        q_values = self.fc_net(conv_out)
        return q_values

    def get_best_action(self, observation):
        if len(observation.shape) == 3:
            observation.unsqueeze_(0)
        with torch.no_grad():
            q_values = self.forward(observation)
        return torch.argmax(q_values, dim=1)

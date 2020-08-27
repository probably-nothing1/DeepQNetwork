# import gin
import gin
import numpy as np
import torch
import torch.nn as nn

from utils.utils import create_conv_network, create_fully_connected_network


def create_dqn_agent(env):
    env_name = env.unwrapped.spec.id
    action_space = env.action_space
    observation_shape = env.observation_space.shape
    if env_name in ["CartPole-v0"]:
        return DeepQNetwork(observation_shape, action_space)
    elif env_name.startswith("Pong"):
        return AtariDeepQNetwork(observation_shape, action_space)

    raise ValueError(f"Can't create DQN model for {env_name} environment")


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
    def sample(self, observation, epsilon):
        if torch.rand(1).item() < epsilon:
            return self.action_space.sample()

        q_values = self(observation)
        return torch.argmax(q_values).item()


@gin.configurable
class AtariDeepQNetwork(DeepQNetwork):
    def __init__(self, observation_shape, action_space, conv_sizes, fc_sizes, use_bn):
        super().__init__(observation_shape, action_space, fc_sizes)
        self.action_space = action_space
        conv_sizes[0][0] = observation_shape[0]
        conv_net = create_conv_network(conv_sizes, use_bn=use_bn)
        fc_input_size = self._compute_conv_out(conv_net, observation_shape)
        fc_net = create_fully_connected_network([fc_input_size, *fc_sizes, self.action_space.n], activation_fn=nn.ReLU)
        self.model = nn.Sequential(conv_net, nn.Flatten(), fc_net)

    @torch.no_grad()
    def _compute_conv_out(self, conv_net, observation_shape):
        dummy_input = torch.zeros(1, *observation_shape)
        out = conv_net(dummy_input)
        return int(np.prod(out.shape))

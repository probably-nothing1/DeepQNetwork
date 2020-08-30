import gin
import numpy as np
import torch
import torch.nn as nn

from models.DeepQNetwork import DeepQNetwork
from utils.utils import create_conv_network, create_fully_connected_network


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

from itertools import chain, zip_longest

import gin
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.nn import BatchNorm2d, Conv2d, Identity, Linear, ReLU, Tanh


@gin.configurable
def setup_logger(name="run-name", notes="", project="project-name", tags=[], save_code=True, monitor_gym=True):
    wandb.init(**locals())


@gin.configurable
def set_seed(env, seed=1337):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)


def create_fully_connected_network(sizes, activation_fn=Tanh, output_activation_fn=Identity):
    fc_layers = [Linear(in_size, out_size) for in_size, out_size in zip(sizes[:-1], sizes[1:])]
    activations = [activation_fn() for _ in range(len(fc_layers) - 1)] + [output_activation_fn()]
    layers = [x for x in chain(*zip_longest(fc_layers, activations)) if x is not None]
    return nn.Sequential(*layers)


def create_conv_network(conv_sizes, activation_fn=ReLU, use_bn=True):
    layers = []
    for in_size, out_size, kernel, stride in conv_sizes:
        layers.append(Conv2d(in_size, out_size, kernel_size=kernel, stride=stride))
        if use_bn:
            layers.append(BatchNorm2d(out_size))
        layers.append(activation_fn())

    return nn.Sequential(*layers)

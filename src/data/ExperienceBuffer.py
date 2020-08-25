import gin
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@gin.configurable
class ExperienceBuffer:
    def __init__(self, size, observation_shape):
        self.index = 0
        self.size = size
        self.observations = np.zeros((size, *observation_shape))
        self.next_observations = np.zeros((size, *observation_shape))
        self.actions = np.zeros(size)
        self.rewards = np.zeros(size)
        self.not_done = np.zeros(size)
        self.is_full = False

    def append(self, observation, action, reward, next_observation, done):
        self.observations[self.index] = observation
        self.next_observations[self.index] = next_observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.not_done[self.index] = not done
        self.index += 1
        if not self.is_full and self.index == self.size:
            self.is_full = True
        self.index %= self.size

    def create_dataloder(self, device):
        dataset = TensorDataset(
            torch.as_tensor(self.observations, dtype=torch.float32, device=device),
            torch.as_tensor(self.actions, dtype=torch.int64, device=device),
            torch.as_tensor(self.rewards, dtype=torch.float32, device=device),
            torch.as_tensor(self.next_observations, dtype=torch.float32, device=device),
            torch.as_tensor(self.not_done, dtype=torch.float32, device=device),
        )
        return DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    def clear(self):
        self.trajectories.clear()

    def is_ready(self):
        return self.is_full

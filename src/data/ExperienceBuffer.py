import gin
import numpy as np
import torch


@gin.configurable
class ExperienceBuffer:
    def __init__(self, observation_shape, capacity):
        self.index = 0
        self.capacity = capacity
        self.observations = np.zeros((capacity, *observation_shape))
        self.next_observations = np.zeros((capacity, *observation_shape))
        self.actions = np.zeros(capacity)
        self.rewards = np.zeros(capacity)
        self.done = np.zeros(capacity)
        self.is_full = False
        self.size = 0

    def append(self, observation, action, reward, next_observation, done):
        self.observations[self.index] = observation
        self.next_observations[self.index] = next_observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done
        self.index += 1
        if self.size < self.capacity:
            self.size += 1
        self.index %= self.capacity

    def get_batch(self, bs, device):
        indices = np.random.choice(self.size, bs, replace=False)
        return (
            torch.as_tensor(self.observations[indices], dtype=torch.float32, device=device),
            torch.as_tensor(self.actions[indices], dtype=torch.int64, device=device),
            torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            torch.as_tensor(self.next_observations[indices], dtype=torch.float32, device=device),
            torch.as_tensor(self.done[indices], dtype=torch.float32, device=device),
        )

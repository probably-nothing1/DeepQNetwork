import copy
import sys
from time import time

import gin
import torch
import wandb
from torch.nn.functional import mse_loss
from torch.optim import Adam

from data.ExperienceBuffer import ExperienceBuffer
from evaluation import evaluate, record_evaluation_video
from models.DeepQNetwork import DeepQNetwork
from utils.env_utils import create_environment
from utils.utils import set_seed, setup_logger

start_epislon = 1.0
end_epsilon = 0.01


def get_epsilon(steps, max_steps):
    new_epislon = start_epislon - (start_epislon - end_epsilon) * steps / max_steps
    return max(new_epislon, end_epsilon)


def train_dqn(deep_q_network, deep_q_network_target, dataloader, dqn_optimizer, gamma=0.99):
    training_start_time = time()
    total_loss = 0
    for o, a, r, next_o, done in dataloader:
        dqn_optimizer.zero_grad()
        q_vals = deep_q_network.get_q_values(o, a)
        max_q_val_target = deep_q_network_target.get_max_q_values(next_o)
        temp = gamma * max_q_val_target
        temp2 = done * temp
        expected_q_val_target = r + temp2
        loss = mse_loss(q_vals, expected_q_val_target)
        loss.backward()
        dqn_optimizer.step()
        total_loss += loss.item()
        break

    wandb.log(
        {"Training Loss": total_loss, "Training FPS": len(dataloader.batch_size) / (time() - training_start_time)}
    )


@gin.configurable
def main(lr, weight_decay, record_eval_video_rate, max_steps, device):
    env = create_environment()

    setup_logger()
    set_seed(env)

    deep_q_network = DeepQNetwork(env.observation_space.shape, env.action_space).to(device)
    deep_q_network_target = DeepQNetwork(env.observation_space.shape, env.action_space).to(device)
    # deep_q_network_target = copy.deepcopy(deep_q_network)
    experience_buffer = ExperienceBuffer(observation_shape=env.observation_space.shape)
    dqn_optimizer = Adam(deep_q_network.parameters(), lr=lr, weight_decay=weight_decay)

    steps_collected = 0
    episode = 0
    while steps_collected < max_steps:
        episode_start_time = time()
        o = env.reset()
        total_reward = 0
        done = False
        episode_steps = 0
        while not done:
            o_tensor = torch.as_tensor(o, dtype=torch.float32).to(device)
            epsilon = get_epsilon(steps_collected, max_steps)
            wandb.log({"Epsilon": epsilon})
            action = deep_q_network.sample(o_tensor, epsilon)

            next_o, reward, done, info = env.step(action)
            total_reward += reward

            experience_buffer.append(o, action, reward, next_o, done)
            o = next_o.copy()
            episode_steps += 1
            steps_collected += 1

            if experience_buffer.is_ready():
                dataloader = experience_buffer.create_dataloder(device)
                train_dqn(deep_q_network, deep_q_network_target, dataloader, dqn_optimizer, gamma=0.99)

        episode += 1
        evaluate(deep_q_network, env, device)
        if episode % record_eval_video_rate == 0:
            record_evaluation_video(deep_q_network, env, device)

        deep_q_network_target.load_state_dict(deep_q_network.state_dict())

        wandb.log(
            {
                "Episode": episode,
                "Simulation Episode Total Reward": total_reward,
                "Simulation Episode Steps": episode_steps,
                "Simulation FPS": episode_steps / (time() - episode_start_time),
            }
        )


# python src/main.py experiments/dev_dqn_config_gpu_0.gin
if __name__ == "__main__":
    experiment_file = sys.argv[1]
    gin.parse_config_file(experiment_file)
    main()

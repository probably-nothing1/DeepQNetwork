import math
import sys
from time import time

import gin
import torch
import wandb
from torch.nn.functional import mse_loss
from torch.optim import Adam

from data.ExperienceBuffer import ExperienceBuffer
from evaluation import evaluate, record_evaluation_video
from models.DeepQNetwork import create_dqn_agent
from utils.env_utils import create_environment
from utils.utils import set_seed, setup_logger


@gin.configurable
def get_epsilon(steps, eps_start=0.9, eps_end=0.02, eps_decay=300):
    eps = eps_end + (eps_start - eps_end) * math.exp(-steps / eps_decay)
    eps = max(eps, eps_end)
    wandb.log({"Epsilon": eps})
    return eps


@gin.configurable
def train_dqn(dqn, dqn_target, experience_buffer, dqn_optimizer, device, gamma, bs):
    if experience_buffer.size < bs:
        return

    training_start_time = time()
    total_loss = 0

    o, a, r, next_o, done = experience_buffer.get_batch(bs, device)
    q_vals = dqn.get_q_values(o, a)

    max_next_q_values = dqn_target.get_max_q_values(next_o)
    q_vals_target = r + (1 - done) * gamma * max_next_q_values

    loss = mse_loss(q_vals, q_vals_target)

    dqn_optimizer.zero_grad()
    loss.backward()
    dqn_optimizer.step()
    total_loss += loss.item()

    wandb.log({"Training Loss": total_loss, "Training FPS": bs / (time() - training_start_time)})


@gin.configurable
def main(lr, weight_decay, record_eval_video_rate, max_steps, device):
    env = create_environment()

    setup_logger()
    set_seed(env)

    dqn = create_dqn_agent(env).to(device)
    dqn_target = create_dqn_agent(env).to(device)
    experience_buffer = ExperienceBuffer(observation_shape=env.observation_space.shape)
    dqn_optimizer = Adam(dqn.parameters(), lr=lr, weight_decay=weight_decay)

    print(dqn)

    steps_collected = 0
    episode = 0
    while steps_collected < max_steps:
        episode_total_time = 0
        o = env.reset()
        total_reward = 0
        done = False
        episode_steps = 0
        while not done:
            start_iter = time()
            o_tensor = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
            epsilon = get_epsilon(steps_collected)
            a = dqn.sample(o_tensor, epsilon)

            next_o, reward, done, info = env.step(a)
            total_reward += reward

            experience_buffer.append(o, a, reward, next_o, done)
            o = next_o.copy()
            episode_steps += 1
            steps_collected += 1
            episode_total_time += time() - start_iter

            train_dqn(dqn, dqn_target, experience_buffer, dqn_optimizer, device=device)

            if steps_collected % 1000 == 0:
                dqn_target.load_state_dict(dqn.state_dict())

        evaluate(dqn, env, device)
        episode += 1
        if episode % record_eval_video_rate == 0:
            record_evaluation_video(dqn, env, device)

        print(
            f"Episde {episode}. Simulation episode reward {total_reward} steps {episode_steps}. FPS: {episode_steps / episode_total_time}"
        )
        wandb.log(
            {
                "Episode": episode,
                "Simulation Episode Total Reward": total_reward,
                "Simulation Episode Steps": episode_steps,
                "Simulation FPS": episode_steps / episode_total_time,
            }
        )


# python src/main.py experiments/dev_dqn_config_gpu_0.gin
# python src/main.py experiments/dev_dqn_config_gpu_1.gin
# xvfb-run -a python src/main.py experiments/test_CartPole_cpu.gin
if __name__ == "__main__":
    experiment_file = sys.argv[1]
    gin.parse_config_file(experiment_file)
    main()

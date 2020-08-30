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
from models import create_dqn_agent
from utils.env_utils import create_environment
from utils.tricks import polyak_averaging
from utils.utils import set_seed, setup_logger


@gin.configurable
def get_epsilon(steps, eps_start=0.9, eps_end=0.02, eps_decay=300):
    eps = eps_end + (eps_start - eps_end) * math.exp(-steps / eps_decay)
    eps = max(eps, eps_end)
    wandb.log({"Epsilon": eps})
    return eps


@gin.configurable
def train_dqn(dqn, dqn_target, experience_buffer, dqn_optimizer, device, gamma, bs, use_double_dqn=False):
    if experience_buffer.size < bs:
        return

    training_start_time = time()
    total_loss = 0

    o, a, r, next_o, done = experience_buffer.get_batch(bs, device)
    q_vals = dqn.get_q_values(o, a)

    if use_double_dqn:
        best_action_look_ahead = dqn.get_max_actions(next_o)
        max_next_q_values = dqn_target.get_q_values(next_o, best_action_look_ahead)
    else:
        max_next_q_values = dqn_target.get_max_q_values(next_o)

    q_vals_target = r + (1 - done) * gamma * max_next_q_values

    loss = mse_loss(q_vals, q_vals_target)

    dqn_optimizer.zero_grad()
    loss.backward()
    dqn_optimizer.step()
    total_loss += loss.item()

    wandb.log({"Training Loss": total_loss, "Training FPS": bs / (time() - training_start_time)})


@gin.configurable
def main(lr, weight_decay, record_eval_video_rate, max_steps, device, early_stop_mean_reward, use_polyak_averaging):
    env = create_environment()

    setup_logger()
    set_seed(env)

    dqn = create_dqn_agent(env).to(device)
    dqn_target = create_dqn_agent(env).to(device)
    experience_buffer = ExperienceBuffer(observation_shape=env.observation_space.shape)
    dqn_optimizer = Adam(dqn.parameters(), lr=lr, weight_decay=weight_decay)

    print(dqn)

    runnig_mean_reward = 0
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

            # if env.unwrapped.spec.id.startswith("Pong") and not done and reward != 0:
            #     # mark termination of sub-game when point is handed. Only Pong envs
            #     experience_buffer.append(o, a, reward, next_o, True)
            #     env.step(0), env.step(0)  # NOOP
            # else:
            #     experience_buffer.append(o, a, reward, next_o, done)
            experience_buffer.append(o, a, reward, next_o, done)

            o = next_o.copy()
            episode_steps += 1
            steps_collected += 1
            episode_total_time += time() - start_iter

            train_dqn(dqn, dqn_target, experience_buffer, dqn_optimizer, device=device)

            if use_polyak_averaging:
                polyak_averaging(dqn_target, dqn)
            elif steps_collected % 1000 == 0:
                dqn_target.load_state_dict(dqn.state_dict())

        # evaluate, early stop and record
        episode += 1
        if episode % record_eval_video_rate == 0:
            record_evaluation_video(dqn, env, device)

        mean_reward, _ = evaluate(dqn, env, device)
        if episode == 1:
            runnig_mean_reward = mean_reward
        else:
            runnig_mean_reward = 0.9 * runnig_mean_reward + 0.1 * mean_reward
        if early_stop_mean_reward < runnig_mean_reward:
            return

        # logging
        fps = int(episode_steps / episode_total_time)
        print(f"Episde {episode} | Reward {total_reward} | Steps {episode_steps} | FPS: {fps}")
        wandb.log(
            {
                "Episode": episode,
                "Simulation Episode Total Reward": total_reward,
                "Simulation Episode Steps": episode_steps,
                "Simulation FPS": fps,
                "Moving Average Test Reward": runnig_mean_reward,
            }
        )


# python src/main.py experiments/dev_dqn_config_gpu_0.gin
# python src/main.py experiments/dev_dqn_config_gpu_1.gin
# xvfb-run -a python src/main.py experiments/test_CartPole_cpu.gin
if __name__ == "__main__":
    experiment_file = sys.argv[1]
    gin.parse_config_file(experiment_file)
    main()

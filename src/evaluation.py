import gin
import gym
import numpy as np
import torch
import wandb

from utils.env_utils import get_video_filepath


def record_evaluation_video(agent, env, device):
    is_recording = isinstance(env, gym.wrappers.Monitor)
    if is_recording:
        env._set_mode("evaluation")
        evaluate_one(agent, env, device)
        env._set_mode("training")
        env.reset()
        video_filepath = get_video_filepath(env)
        wandb.log({"Evaluate Video": wandb.Video(video_filepath)})


def evaluate_one(agent, env, device):
    total_reward = 0
    o = env.reset()
    done = False
    while not done:
        o = torch.as_tensor(o, dtype=torch.float32).to(device)

        best_action = agent.get_best_action(o)
        next_o, reward, done, info = env.step(best_action.cpu().numpy())

        total_reward += reward
        o = next_o

    return total_reward


@gin.configurable
def evaluate(agent, env, device, runs=1):
    total_rewards = np.zeros(runs)
    for i in range(runs):
        total_rewards[i] = evaluate_one(agent, env, device)

    return total_rewards.mean(), total_rewards.max(), total_rewards.min()

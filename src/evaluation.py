import gin
import gym
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
    steps = 0
    o = env.reset()
    done = False
    while not done:
        o = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device)

        best_action = agent.sample(o, epsilon=0)
        next_o, reward, done, info = env.step(best_action)
        o = next_o

        total_reward += reward
        steps += 1

    return total_reward, steps


@gin.configurable
def evaluate(agent, env, device):
    total_reward, steps = evaluate_one(agent, env, device)
    wandb.log({"Test Steps": steps, "Test Reward": total_reward})
    return total_reward, steps

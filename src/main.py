import gin
import torch
import wandb
from torch.optim import Adam

from data.ExperienceBuffer import ExperienceBuffer
from evaluation import evaluate, record_evaluation_video
from models.DeepQNetwork import DeepQNetwork
from utils.env_utils import create_environment
from utils.utils import dict_iter2tensor, set_seed, setup_logger


def train_dqn(deep_q_network, data, dqn_optimizer):
    pass


def pick_action(q):
    pass


@gin.configurable
def collect_trajectories(actor, env, experience_buffer, min_num_of_steps_in_epoch):
    steps_collected = 0
    while steps_collected < min_num_of_steps_in_epoch:
        o = env.reset()
        total_reward = 0
        done = False
        while not done:
            o = torch.as_tensor(o, dtype=torch.float32)
            with torch.no_grad():
                q_values, _ = actor(o)

            q_values = q_values.numpy()
            action = pick_action(q_values)
            next_o, reward, done, info = env.step(action)
            total_reward += reward

            experience_buffer.append(o, q_values, action, reward, done)
            o = next_o
            steps_collected += 1

        print(f"Total Reward {total_reward}")
        wandb.log({"Total Reward": total_reward})

    print(f"steps collected {steps_collected}")


@gin.configurable
def main(lr, weight_decay, epochs, record_eval_video_rate, device="cpu"):
    env = create_environment()
    # observation_dim = env.observation_space.shape[0]

    setup_logger()
    set_seed(env)

    deep_q_network = DeepQNetwork(env.observation_space.shape, env.action_space.n).to(device)
    # experience_buffer = ExperienceBuffer()
    # dqn_optimizer = Adam(deep_q_network.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        print(f"epoch {epoch}")
        # collect_trajectories(deep_q_network, env, experience_buffer)

        # data = experience_buffer.get_data()
        # data = dict_iter2tensor(data)
        # dqn_loss, entropy = train_dqn(deep_q_network, data, dqn_optimizer)
        # experience_buffer.clear()

        if epoch % record_eval_video_rate == 0:
            record_evaluation_video(deep_q_network, env, device)
        test_mean_reward, test_max_reward, test_min_reward = evaluate(deep_q_network, env, device)

        wandb.log(
            {
                # "DQN Loss": actor_loss,
                # "Entropy": entropy,
                "Test Mean Reward": test_mean_reward,
                "Test Max Reward": test_max_reward,
                "Test Min Reward": test_min_reward,
            }
        )


if __name__ == "__main__":
    gin.parse_config_file("experiments/dev_dqn_config.gin")
    main()

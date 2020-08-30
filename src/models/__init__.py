from models.AtariDeepQNetwork import AtariDeepQNetwork
from models.DeepQNetwork import DeepQNetwork


def create_dqn_agent(env):
    env_name = env.unwrapped.spec.id
    action_space = env.action_space
    observation_shape = env.observation_space.shape
    if env_name in ["CartPole-v0"]:
        return DeepQNetwork(observation_shape, action_space)
    elif env_name.startswith("Pong"):
        return AtariDeepQNetwork(observation_shape, action_space)

    raise ValueError(f"Can't create DQN model for {env_name} environment")

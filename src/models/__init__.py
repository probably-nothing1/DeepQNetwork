import gin

from models.AtariDeepQNetwork import AtariDeepQNetwork
from models.DeepQNetwork import DeepQNetwork


@gin.configurable
def create_dqn_agent(env, advantage=False):
    env_name = env.unwrapped.spec.id
    action_space = env.action_space
    observation_shape = env.observation_space.shape
    if env_name in ["CartPole-v0"]:
        return DeepQNetwork(observation_shape, action_space, advantage=advantage)
    elif env_name.startswith("Pong"):
        return AtariDeepQNetwork(observation_shape, action_space, advantage=advantage)

    raise ValueError(f"Can't create DQN model for {env_name} environment")

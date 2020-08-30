# DeepQNetwork
DQN model with various tricks and improvements

DONE:
* working on CartPole and Pong
* DQN with target network
* double DQN
* tweak wrappers so every point is one game in pong
* Advantage learning (Dueling DQN)

IN PROGRESS:
* Polayk averaging (not working ?)

TODO algo trick:
* n-step q-learning
* noisy network
* prioritized buffer
* clipped double q learning ?
* Categorical DQN

TODO perf tricks:
* move ExperienceBuffer to GPU (? idk if that helps)
* paralle simulation and training
* many envs in parallel (?)
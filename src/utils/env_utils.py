import os

import cv2
import gin
import gym
import numpy as np
from gym import wrappers


def get_video_filepath(env):
    assert isinstance(env, wrappers.Monitor)

    def pad_episode_number(id):
        return str(1000000 + id)[1:]

    episode_number = pad_episode_number(env.episode_id - 2)
    filepath = f"{env.file_prefix}.video.{env.file_infix}.video{episode_number}.mp4"
    return os.path.join(env.directory, filepath)


class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, img):
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.597 + img[:, :, 2] * 0.114
        img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        img = img[18:102, :]
        return np.reshape(img, [84, 84, 1])


class ScaleImage(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        return np.array(observation, dtype=np.float32) / 255.0


class ImageToPytorchChannelOrdering(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], *old_shape[:-1])
        self.observation_space = gym.spaces.Box(0.0, 1.0, new_shape)

    def observation(self, observation):
        return np.einsum("hwc->chw", observation)


class EvalMonitor(wrappers.Monitor):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.mode = "training"

    def _set_mode(self, mode):
        assert mode in ["evaluation", "training"], "Invalid mode"
        self.mode = mode
        self.stats_recorder.type = "t" if mode == "training" else "e"

    def _video_enabled(self):
        return self.video_callable(self.mode)


@gin.configurable
def create_environment(name, gym_make_kwargs=dict(), save_videos=False, wrapper_kwargs=dict()):
    env = gym.make(name, **gym_make_kwargs)
    env = ProcessFrame(env)
    env = ScaleImage(env)
    env = ImageToPytorchChannelOrdering(env)
    if save_videos:
        env = EvalMonitor(env, video_callable=lambda mode: mode == "evaluation", **wrapper_kwargs)
        # env = wrappers.Monitor(env, **wrapper_kwargs)
        # env = wrappers.Monitor(env, "./videos/" + str(time.time()) + "/", force=True, write_upon_reset=True)
    return env

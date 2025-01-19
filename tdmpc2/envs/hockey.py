import torch

from envs.wrappers.timeout import Timeout
from hockey import hockey_env


class HockeyEnvAdapter(hockey_env.HockeyEnv_BasicOpponent):
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        info["success"] = info.get("winner", 0) == 1
        return obs, reward, done or truncated, info

    def reset(self, one_starting=None, mode=None, seed=None, options=None):
        (obs, _) = super().reset(one_starting, mode, seed, options)
        return torch.from_numpy(obs).float()


def make_env(cfg):
    if cfg.task != "hockey":
        raise ValueError("Unknown task:", cfg.task)

    env = HockeyEnvAdapter()
    env = Timeout(env, max_episode_steps=500)

    return env

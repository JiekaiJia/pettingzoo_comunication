"""This module provides functions that make sure environment to be compatible with RLlib. If Rllib is not used, please
directly use the wrapper in comm_channel.py."""

import numpy as np
from ray.rllib.env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from supersuit import pad_action_space_v0, pad_observations_v0

from comm_channel import ParallelCommWrapper, CommWrapper


def main_comm_env(base_env, comm_bits=5):
    """Wrap the communication channel into Pettingzoo main environment, and padding the environment."""
    def comm_env(**kwargs):
        env = CommWrapper(base_env, comm_bits, **kwargs)
        env = pad_observations_v0(env)
        env = pad_action_space_v0(env)
        env = _PettingZooEnv(env)
        return env
    return comm_env


def main_env(base_env):
    """Padding the environment."""
    def env(**kwargs):
        env = base_env.env(**kwargs)
        env = pad_observations_v0(env)
        env = pad_action_space_v0(env)
        env = _PettingZooEnv(env)
        return env
    return env


def parallel_comm_env(base_env, comm_bits=5):
    """Wrap the communication channel into Pettingzoo parallel environment, and padding the environment."""
    def comm_env(**kwargs):
        env = ParallelCommWrapper(base_env, comm_bits, **kwargs)
        env = pad_observations_v0(env)
        env = pad_action_space_v0(env)
        env = _ParallelPettingZooEnv(env)
        return env
    return comm_env


def parallel_env(base_env):
    """Padding the parallel environment."""
    def env(**kwargs):
        env = base_env.parallel_env(**kwargs)
        env = pad_observations_v0(env)
        env = pad_action_space_v0(env)
        env = _ParallelPettingZooEnv(env)
        return env
    return env


class _PettingZooEnv(PettingZooEnv):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action_dict):
        # Ensure the input actions are discrete number.
        for k, v in action_dict.items():
            if isinstance(v, (np.int64, np.int32, np.int16, np.int8, int)):
                pass
            elif not v:
                pass
            else:
                action_dict[k] = np.argmax(v)
        return super().step(action_dict)


class _ParallelPettingZooEnv(ParallelPettingZooEnv):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action_dict):
        # Ensure the input actions are discrete number.
        for k, v in action_dict.items():
            if isinstance(v, (np.int64, np.int32, np.int16, np.int8, int)):
                pass
            else:
                action_dict[k] = np.argmax(v)
        return super().step(action_dict)

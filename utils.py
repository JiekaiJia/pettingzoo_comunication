""""""

from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from supersuit import pad_action_space_v0, pad_observations_v0

from comm_channel import ParallelCommWrapper


def parallel_comm_env(base_env, comm_bits=2):
    def comm_env(**kwargs):
        env = ParallelCommWrapper(base_env, comm_bits, **kwargs)
        env = pad_observations_v0(env)
        env = pad_action_space_v0(env)
        env = ParallelPettingZooEnv(env)
        return env
    return comm_env


def parallel_env(base_env):
    def env(**kwargs):
        env = base_env.parallel_env(**kwargs)
        env = pad_observations_v0(env)
        env = pad_action_space_v0(env)
        env = ParallelPettingZooEnv(env)
        return env
    return env
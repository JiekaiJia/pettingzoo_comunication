""""""
from pettingzoo.mpe import (
    simple_crypto_v2,
    simple_reference_v2,
    simple_speaker_listener_v3,
    simple_spread_v2,
    simple_tag_v2,
    simple_world_comm_v2,
)
from ray.tune.registry import register_env

from utils import parallel_env, parallel_comm_env, main_comm_env, main_env


def env():
    return parallel_comm_env(simple_spread_v2)()

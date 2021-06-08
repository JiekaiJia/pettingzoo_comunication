from gym import spaces
import numpy as np
from scipy.special import softmax
from pettingzoo.mpe import simple_spread_v2, simple_speaker_listener_v3, simple_tag_v2, simple_reference_v2, simple_world_comm_v2,simple_crypto_v2
from pettingzoo.utils.conversions import to_parallel_wrapper
from pettingzoo.utils.wrappers import AssertOutOfBoundsWrapper, OrderEnforcingWrapper
from comm_channel import ParallelCommWrapper, CommWrapper
from ray.rllib.env import PettingZooEnv
import random


# env = CommWrapper(simple_crypto_v2, comm_bits=4)
# env.reset()
# for agent in env.agent_iter():
#     # env.render()
#     observations, rewards, dones, infos = env.last()
#     print(agent)
#     print(observations)
#     # print(rewards)
#     if dones:
#         action = None
#     else:
#         action = random.choice([0, 5, 10, 15])
#
#     env.step(action)

par_env = ParallelCommWrapper(simple_crypto_v2, comm_bits=4)
obs = par_env.reset()
for step in range(25):
    print(obs)
    actions = {agent: random.choice([0, 5, 10, 15]) for agent in par_env.agents}
    obs, _, _, _ = par_env.step(actions)

from gym import spaces
import numpy as np
from scipy.special import softmax
from pettingzoo.mpe import simple_spread_v2, simple_speaker_listener_v3, simple_tag_v2, simple_reference_v2, simple_world_comm_v2,simple_crypto_v2
from pettingzoo.sisl import multiwalker_v7, pursuit_v3, waterworld_v3
from pettingzoo.utils.conversions import to_parallel_wrapper
from pettingzoo.utils.wrappers import AssertOutOfBoundsWrapper, OrderEnforcingWrapper
from comm_channel import ParallelCommWrapper, CommWrapper
from ray.rllib.env import PettingZooEnv
from ray.rllib.policy.tf_policy_template import build_tf_policy
import random

from utils import init_comm_dict

#
# env = multiwalker_v7.env()
# comm_dict = init_comm_dict(env)
# comm_dict['comm_bits'] = 4
# comm_dict['receivers'][env.possible_agents[2]] = [env.possible_agents[0], env.possible_agents[1]]
# env = CommWrapper(env, comm_dict)
# env.reset()
# for agent in env.agent_iter():
#     env.render()
#     observations, rewards, dones, infos = env.last()
#     # print(rewards)
#     if dones:
#         action = None
#     else:
#         # action = random.choice([0, 5, 10, 15])
#         action = np.tanh(np.random.randn(1, env.action_spaces[agent].shape[0]).reshape(env.action_spaces[agent].shape[0],))
#     env.step(action)


# par_env = multiwalker_v7.parallel_env()
# comm_dict = init_comm_dict(par_env)
# comm_dict['comm_bits'] = 4
# comm_dict['receivers'][par_env.possible_agents[2]] = [par_env.possible_agents[0], par_env.possible_agents[1]]
# par_env = ParallelCommWrapper(par_env, comm_dict)
# obs = par_env.reset()
# for step in range(500):
#     par_env.render()
#     # actions = {agent: random.choice([0, 5, 10, 15]) for agent in par_env.agents}
#     actions = {agent: np.tanh(np.random.randn(1, par_env.action_spaces[agent].shape[0]).reshape(par_env.action_spaces[agent].shape[0],)) for agent in par_env.agents}
#     obs, _, _, _ = par_env.step(actions)

tmp = list((0, 1, 2))
print(np.array(tmp).shape)
tmp[-1] += 1
v = tuple(tmp)
print(v)

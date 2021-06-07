from gym import spaces
import numpy as np
from scipy.special import softmax
from pettingzoo.mpe import simple_spread_v2, simple_speaker_listener_v3, simple_tag_v2, simple_reference_v2, simple_world_comm_v2
from pettingzoo.utils.conversions import to_parallel_wrapper
from pettingzoo.utils.wrappers import AssertOutOfBoundsWrapper, OrderEnforcingWrapper
from utils import mpe_parallel_env

env = mpe_parallel_env(simple_speaker_listener_v3)
obs = env.reset()
# print(obs)
for step in range(25):
    # env.render()
    actions = {agent: np.random.randint(0, env.action_spaces[agent].n) for agent in env.agents}
    observations, rewards, dones, infos = env.step(actions)
    # print(observations)





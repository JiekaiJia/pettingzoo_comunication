import os
import gym
from gym.spaces import Discrete, Box
from pettingzoo.mpe import simple_spread_v2
import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import shutil


# env = simple_spread_v2.env(N=3, local_ratio=0.5, max_cycles=100)
#
# env.reset()
# for agent in env.agent_iter():
#     env.render()
#     observation, reward, done, info = env.last()
#     if not done:
#         action = env.action_spaces[agent].sample()
#     else:
#         action = None
#     env.step(action)

ray.shutdown()
ray.init(ignore_reinit_error=True)

CHECKPOINT_ROOT = "tmp/ppo/taxi"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

ray_results = os.getenv("HOME") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

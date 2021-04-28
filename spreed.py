import os
import gym
from gym.spaces import Discrete, Box
from pettingzoo.mpe import simple_spread_v2
import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
import shutil


env = simple_spread_v2.env(max_cycles = 100)

env.reset()
for agent in env.agent_iter():
    env.render()
    observation, reward, done, info = env.last()
    if not done:
        action = env.action_spaces[agent].sample()
    else:
        action = None
    env.step(action)

ray.shutdown()
info = ray.init(ignore_reinit_error=True)
print(info)

CHECKPOINT_ROOT = "tmp/ppo/taxi"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

ray_results = os.getenv("HOME") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

config = DEFAULT_CONFIG.copy()
config['num_workers'] = 1
config['num_sgd_iter'] = 30
config['sgd_minibatch_size'] = 128
config['model']['fcnet_hiddens'] = [100, 100]
config['num_cpus_per_worker'] = 0

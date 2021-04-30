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


env = simple_spread_v2.env(max_cycles=10)

env.reset()
last_reward = 0
for agent in env.agent_iter():
    env.render()
    done = env.dones[agent]
    current_reward = env.rewards[agent]
    if not done:
        if current_reward > last_reward:
            pass
        else:
            action = env.action_spaces[agent].sample()
    else:
        action = None

    observation, reward, _, info = env.last()
    last_reward = reward
    print(done)
    env.step(action)


ray.shutdown()

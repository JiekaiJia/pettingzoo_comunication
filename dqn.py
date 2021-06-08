""""""
from copy import deepcopy
import os

from pettingzoo.mpe import (
    simple_crypto_v2,
    simple_reference_v2,
    simple_speaker_listener_v3,
    simple_spread_v2,
    simple_tag_v2,
    simple_world_comm_v2,
)
import ray
from ray.rllib.agents.registry import get_trainer_class
from ray import tune
from ray.tune.registry import register_env

from utils import parallel_env, parallel_comm_env, main_comm_env, main_env


if __name__ == '__main__':
    # Create test environment.
    test_env = main_env(simple_speaker_listener_v3)()
    # Register env
    register_env('simple_spread', lambda _: main_env(simple_speaker_listener_v3)())

    # The used algorithm
    alg_name = 'DQN'
    # Gets default training configuration.
    config = deepcopy(get_trainer_class(alg_name)._default_config)

    # === Settings for Rollout Worker processes ===
    # Use GPUs if `RLLIB_NUM_GPUS` env var set to > 0.
    config['num_gpus'] = int(os.environ.get('RLLIB_NUM_GPUS', '0'))
    # Number of rollout worker actors to create for parallel sampling.
    config['num_workers'] = 1
    config['num_envs_per_worker'] = 4
    config['render_env'] = True

    # === Settings for the Trainer process ===
    # Whether layers should be shared for the value function.
    config['model'] = {
        # 'fcnet_hiddens': [128],
        'fcnet_activation': 'relu',
    }

    # === Environment Settings ===
    config['env'] = 'simple_spread'
    # the env_creator function via the register env lambda below.
    # config['env_config'] = {'max_cycles': max_cycles, 'num_agents': num_agents, 'local_ratio': local_ratio}

    # # === Debug Settings ===
    # # Periodically print out summaries of relevant internal dataflow(DEBUG, INFO, WARN, or ERROR.)
    config['log_level'] = 'DEBUG'

    # === Settings for Multi-Agent Environments ===
    # Configuration for multi-agent setup with policy sharing:
    config['multiagent'] = {
        # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
        # of (policy_cls, obs_space, act_space, config). This defines the
        # observation and action spaces of the policies and any extra config.
        'policies': {
            f'{agent}': (None, test_env.observation_spaces[agent], test_env.action_spaces[agent], {}) for agent in test_env.agents
        },
        # Function mapping agent ids to policy ids.
        'policy_mapping_fn': lambda agent_id: f'{agent_id}',
    }

    # Initialize ray and trainer object
    ray.init(
        ignore_reinit_error=True,
        # log_to_driver=False
    )

    # Stop criteria
    stop = {
        # "episode_reward_mean": -115,
        "training_iteration": 700,
    }

    # Train
    results = tune.run(
        alg_name,
        stop=stop,
        config=config,
        checkpoint_at_end=True,
        checkpoint_freq=1,
        local_dir='./ray_results',
        # restore='/home/jiekaijia/PycharmProjects/pettingzoo_comunication/ray_results/DQN/DQN_simple_spread_91f8e_00000_0_2021-06-07_14-33-32/checkpoint_000008/checkpoint-8',
        num_samples=1
    )

    # Get the tuple of checkpoint_path and metric
    checkpoints = results.get_trial_checkpoints_paths(
        trial=results.get_best_trial("episode_reward_mean", mode="max"),
        metric="episode_reward_mean"
    )

    ray.shutdown()

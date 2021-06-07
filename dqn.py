""""""
from copy import deepcopy
import os

import numpy as np
from pettingzoo.mpe import (
    simple_crypto_v2,
    simple_reference_v2,
    simple_speaker_listener_v3,
    simple_spread_v2,
    simple_tag_v2,
    simple_world_comm_v2,
)
import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray import tune
from ray.tune.registry import register_env
from typing import Dict

from model_register import register_dqn, custom_model
from utils import parallel_env, parallel_comm_env

PRIO_WEIGHTS = "weights"
if __name__ == '__main__':
    class MyCallback(DefaultCallbacks):
        def on_postprocess_trajectory(
                        self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
                        agent_id: str, policy_id: str, policies: Dict[str, Policy],
                        postprocessed_batch: SampleBatch,
                        original_batches: Dict[str, SampleBatch], **kwargs):
            postprocessed_batch["next_actions"] = np.concatenate(
                [postprocessed_batch["actions"][1:],
                 np.zeros_like([policies[policy_id].action_space.sample()])])


    MYDQNTFPolicy = register_dqn()
    custom_model()

    # Create test environment.
    test_env = parallel_comm_env(simple_spread_v2)()
    # Register env
    register_env('simple_spread', lambda _: parallel_comm_env(simple_spread_v2)())

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
        # 'fcnet_activation': 'relu',
        'custom_model': 'keras_q_model',
    }

    # === Environment Settings ===
    config['env'] = 'simple_spread'
    # the env_creator function via the register env lambda below.
    # config['env_config'] = {'max_cycles': max_cycles, 'num_agents': num_agents, 'local_ratio': local_ratio}
    # Discount factor of the MDP.
    config['gamma'] = 0.99

    # === Debug Settings ===
    # Periodically print out summaries of relevant internal dataflow(DEBUG, INFO, WARN, or ERROR.)
    # config['log_level'] = 'DEBUG'

    # === Deep Learning Framework Settings ===
    config['framework'] = 'tf'
    config['callbacks'] = MyCallback

    # === Settings for Multi-Agent Environments ===
    # Configuration for multi-agent setup with policy sharing:
    config['multiagent'] = {
        # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
        # of (policy_cls, obs_space, act_space, config). This defines the
        # observation and action spaces of the policies and any extra config.
        'policies': {
            f'{agent}': (MYDQNTFPolicy, test_env.observation_spaces[agent], test_env.action_spaces[agent], {}) for agent in test_env.agents
        },
        # Function mapping agent ids to policy ids.
        'policy_mapping_fn': lambda agent_id: f'{agent_id}',
    }

    # DQN-specific configs
    # === Model ===
    # N-step Q learning
    config['n_step'] = 3
    config['noisy'] = True
    config['num_atoms'] = 2
    config['v_min'] = -10
    config['v_max'] = 10
    config['hiddens'] = [128]
    config['prioritized_replay'] = True
    config['prioritized_replay_alpha'] = 0.5
    config['final_prioritized_replay_beta'] = 1
    config['prioritized_replay_beta_annealing_timesteps'] = 400000



    # === Optimization ===
    # How many steps of the model to sample before learning starts.
    config['learning_starts'] = 1
    # Learning rate for adam optimizer.
    config['lr'] = 0.001
    # Divide episodes into fragments of this many steps from each worker and for each agent during rollouts!
    config['rollout_fragment_length'] = 25
    # Training batch size -> Fragments are concatenated up to this point.
    config['train_batch_size'] = 1024

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    config['buffer_size'] = int(1e6)

    # === Exploration Settings ===
    # Switch to greedy actions in evaluation workers.
    # config['evaluation_interval'] = 1
    # config['evaluation_num_episodes'] = 20
    # config['evaluation_config'] = {
    #     'explore': False,
    #     # If True, try to render the environment on the local worker or on worker 1
    #     'render_env': True,
    #     # If True, stores videos in this relative directory inside the default
    #     'record_env': 'Videos'
    # }
    # config['explore'] = False
    config['exploration_config'] = {
        # The Exploration class to use.
        'type': 'EpsilonGreedy',
        # Config for the Exploration class' constructor:
        'initial_epsilon': 1.0,
        'epsilon_timesteps': 1000,  # Timesteps over which to anneal epsilon.
        'final_epsilon': 0
    }

    # Initialize ray and trainer object
    ray.init(
        ignore_reinit_error=True,
        # log_to_driver=False
    )

    # Stop criteria
    stop = {
        # "episode_reward_mean": -115,
        "training_iteration": 1400,
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

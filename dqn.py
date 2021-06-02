""""""
from copy import deepcopy
import os

from gym.spaces import Discrete, MultiDiscrete, Box
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.dqn.dqn_tf_policy import compute_q_values, QLoss
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.policy.policy import Policy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.typing import TensorType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.tf_ops import one_hot
import numpy as np
from numpy import float32
from pettingzoo.mpe import simple_spread_v2
import ray
from ray.rllib.agents.dqn import DQNTorchPolicy, DQNTFPolicy
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.env import PettingZooEnv
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.examples.models.rnn_model import RNNModel, TorchRNNModel
from ray.rllib.policy.view_requirement import ViewRequirement
import tensorflow as tf
from supersuit import dtype_v0
import torch
from typing import Dict
import argparse
import numpy as np
import os

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID
from typing import Dict, Optional, TYPE_CHECKING

PRIO_WEIGHTS = "weights"
if __name__ == '__main__':
    """For this script, you need:
    1. Algorithm name and according module, e.g.: 'DQN' + agents.dqn as agent
    2. Name of the aec game you want to train on, e.g.: 'simple_spread'.
    3. num_cpus
    """


    def get_distribution_inputs_and_class(policy: Policy,
                                          model: ModelV2,
                                          input_dict: dict,
                                          *,
                                          explore=True,
                                          **kwargs):
        q_vals = compute_q_values(
            policy, model, input_dict, state_batches=None, explore=explore)
        q_vals = q_vals[0] if isinstance(q_vals, tuple) else q_vals

        policy.q_values = q_vals
        policy.q_func_vars = model.variables()
        return policy.q_values, Categorical, []  # state-out


    def build_q_losses(policy: Policy, model, _,
                       train_batch: SampleBatch) -> TensorType:
        """Constructs the loss for DQNTFPolicy.

        Args:
            policy (Policy): The Policy to calculate the loss for.
            model (ModelV2): The Model to calculate the loss for.
            train_batch (SampleBatch): The training data.

        Returns:
            TensorType: A single loss tensor.
        """
        config = policy.config
        # q network evaluation
        q_t, q_logits_t, q_dist_t, _ = compute_q_values(
            policy,
            model,
            {
                "obs": train_batch[SampleBatch.CUR_OBS],
                'actions': train_batch[SampleBatch.ACTIONS],
                'prev_actions': train_batch[SampleBatch.PREV_ACTIONS],
                'agent_index': train_batch[SampleBatch.AGENT_INDEX],
             },
            state_batches=None,
            explore=False
        )

        # target q network evalution
        q_tp1, q_logits_tp1, q_dist_tp1, _ = compute_q_values(
            policy,
            policy.target_q_model,
            {
                "obs": train_batch[SampleBatch.CUR_OBS],
                'actions': train_batch[SampleBatch.ACTIONS],
                'prev_actions': train_batch[SampleBatch.PREV_ACTIONS],
                'agent_index': train_batch[SampleBatch.AGENT_INDEX],
            },
            state_batches=None,
            explore=False
        )
        if not hasattr(policy, "target_q_func_vars"):
            policy.target_q_func_vars = policy.target_q_model.variables()

        # q scores for actions which we know were selected in the given state.
        one_hot_selection = tf.one_hot(
            tf.cast(train_batch[SampleBatch.ACTIONS], tf.int32),
            policy.action_space.n)
        q_t_selected = tf.reduce_sum(q_t * one_hot_selection, 1)
        q_logits_t_selected = tf.reduce_sum(
            q_logits_t * tf.expand_dims(one_hot_selection, -1), 1)

        # compute estimate of best possible value starting from state at t + 1
        if config["double_q"]:
            q_tp1_using_online_net, q_logits_tp1_using_online_net, \
            q_dist_tp1_using_online_net, _ = compute_q_values(
                policy, model,
                {
                    "obs": train_batch[SampleBatch.NEXT_OBS],
                    'actions': train_batch['next_actions'],
                    'prev_actions': train_batch[SampleBatch.ACTIONS],
                    'agent_index': train_batch[SampleBatch.AGENT_INDEX],
                },
                state_batches=None,
                explore=False)
            q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
            q_tp1_best_one_hot_selection = tf.one_hot(q_tp1_best_using_online_net,
                                                      policy.action_space.n)
            q_tp1_best = tf.reduce_sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
            q_dist_tp1_best = tf.reduce_sum(
                q_dist_tp1 * tf.expand_dims(q_tp1_best_one_hot_selection, -1), 1)
        else:
            q_tp1_best_one_hot_selection = tf.one_hot(
                tf.argmax(q_tp1, 1), policy.action_space.n)
            q_tp1_best = tf.reduce_sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
            q_dist_tp1_best = tf.reduce_sum(
                q_dist_tp1 * tf.expand_dims(q_tp1_best_one_hot_selection, -1), 1)

        policy.q_loss = QLoss(
            q_t_selected, q_logits_t_selected, q_tp1_best, q_dist_tp1_best,
            train_batch[PRIO_WEIGHTS], train_batch[SampleBatch.REWARDS],
            tf.cast(train_batch[SampleBatch.DONES],
                    tf.float32), config["gamma"], config["n_step"],
            config["num_atoms"], config["v_min"], config["v_max"])

        return policy.q_loss.loss


    MYDQNTFPolicy = DQNTFPolicy.with_updates(
        name="MYDQNTFPolicy",
        action_distribution_fn=get_distribution_inputs_and_class,
        loss_fn=build_q_losses,
    )


    class MyKerasQModel(DistributionalQTFModel):
        """Custom model for DQN."""

        def __init__(self, obs_space, action_space, num_outputs, model_config,
                     name, **kw):
            super(MyKerasQModel, self).__init__(
                obs_space, action_space, num_outputs, model_config, name, **kw)

            # Define the core model layers which will be used by the other
            # output heads of DistributionalQModel
            self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(SampleBatch.ACTIONS, space=self.action_space, shift=-1)
            self.view_requirements[SampleBatch.ACTIONS] = ViewRequirement(SampleBatch.ACTIONS, space=self.action_space, shift=0)
            self.view_requirements["next_actions"] = ViewRequirement(SampleBatch.ACTIONS, space=action_space)
            self.view_requirements[SampleBatch.AGENT_INDEX] = ViewRequirement()

            if isinstance(action_space, Discrete):
                self.action_dim = action_space.n
            elif isinstance(action_space, MultiDiscrete):
                self.action_dim = np.sum(action_space.nvec)
            elif action_space.shape is not None:
                self.action_dim = int(np.product(action_space.shape))
            else:
                self.action_dim = int(len(action_space))

            # Add prev-action/reward nodes to input to GRU.
            self.inputs_dim = int(np.product(self.obs_space.shape))
            self.inputs_dim += self.action_dim
            self.inputs_dim += self.action_dim
            self.inputs_dim += 3

            # Define input layers.
            input_layer = tf.keras.layers.Input(shape=self.inputs_dim, name="inputs")
            bn_input1 = tf.keras.layers.BatchNormalization()(input_layer)
            layer1 = tf.keras.layers.Dense(
                128,
                name="my_layer1",
                activation=tf.nn.relu,
                kernel_initializer=normc_initializer(1.0))(bn_input1)
            bn_input2 = tf.keras.layers.BatchNormalization()(layer1)
            # tf.keras.layers.Dropout()
            layer2 = tf.keras.layers.Dense(
                128,
                name="my_layer2",
                activation=tf.nn.relu,
                kernel_initializer=normc_initializer(1.0))(bn_input2)
            bn_input3 = tf.keras.layers.BatchNormalization()(layer2)
            layer_out = tf.keras.layers.Dense(
                num_outputs,
                name="my_out",
                activation=tf.nn.relu,
                kernel_initializer=normc_initializer(1.0))(bn_input3)
            self.base_model = tf.keras.Model(input_layer, layer_out)

        # Implement the core forward method.
        def forward(self, input_dict, state, seq_lens):
            a = input_dict['actions']
            prev_a = input_dict['prev_actions']
            idx = input_dict['agent_index']
            obs = input_dict["obs"]

            input_flatten = [obs]
            if isinstance(self.action_space, (Discrete, MultiDiscrete)):
                a = one_hot(a, self.action_space)
                prev_a = one_hot(prev_a, self.action_space)
            idx = one_hot(tf.cast(idx, tf.int32), Discrete(3))
            input_flatten.append(tf.reshape(tf.cast(a, tf.float32), [-1, self.action_dim]))
            input_flatten.append(tf.reshape(tf.cast(prev_a, tf.float32), [-1, self.action_dim]))
            input_flatten.append(tf.reshape(tf.cast(idx, tf.float32), [-1, 3]))
            input_flatten = tf.concat(input_flatten, axis=1)
            model_out = self.base_model(input_flatten)
            return model_out, state


    # Register the above custom model.
    ModelCatalog.register_custom_model("keras_q_model", MyKerasQModel)


    class MyCallback(DefaultCallbacks):
        def on_postprocess_trajectory(
                        self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
                        agent_id: str, policy_id: str, policies: Dict[str, Policy],
                        postprocessed_batch: SampleBatch,
                        original_batches: Dict[str, SampleBatch], **kwargs):
            postprocessed_batch["next_actions"] = np.concatenate(
                [postprocessed_batch["actions"][1:],
                 np.zeros_like([policies[policy_id].action_space.sample()])])


    # The used algorithm
    alg_name = 'DQN'
    # Environment parameters
    num_agents = 3
    local_ratio = 0.5
    max_cycles = 25

    # # function that outputs the environment you wish to register.
    # def env_creator(config):
    #     my_env = simple_spread_v2.env(
    #         max_cycles=config.get('max_cycles', max_cycles),
    #         N=config.get('num_agents', num_agents),
    #         local_ratio=config.get('local_ratio', local_ratio)
    #     )
    #     my_env = dtype_v0(my_env, dtype=float32)
    #     # env = color_reduction_v0(env, mode='R')
    #     # env = normalize_obs_v0(env)
    #     return my_env


    def mpe_env(base_env):
        env = base_env.parallel_env()
        env = ParallelPettingZooEnv(env)
        return env

    # Gets default training configuration.
    config = deepcopy(get_trainer_class(alg_name)._default_config)

    # Create test environment.
    test_env = mpe_env(simple_spread_v2)

    # Extract space dimensions
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    # === Settings for Rollout Worker processes ===
    # Use GPUs if `RLLIB_NUM_GPUS` env var set to > 0.
    config['num_gpus'] = int(os.environ.get('RLLIB_NUM_GPUS', '0'))
    # Number of rollout worker actors to create for parallel sampling.
    config['num_workers'] = 3
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
    config['env_config'] = {'max_cycles': max_cycles, 'num_agents': num_agents, 'local_ratio': local_ratio}
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
            # f'agent_{i}': (DQNTorchPolicy, obs_space, act_space, {}) for i in range(num_agents)
            'same': (MYDQNTFPolicy, obs_space, act_space, {})
        },
        # Function mapping agent ids to policy ids.
        'policy_mapping_fn': lambda agent_id: 'same',
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
    config['learning_starts'] = 10000
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
    config['evaluation_interval'] = 1
    config['evaluation_num_episodes'] = 20
    config['evaluation_config'] = {
        'explore': False,
        # If True, try to render the environment on the local worker or on worker 1
        'render_env': True,
        # If True, stores videos in this relative directory inside the default
        'record_env': 'Videos'
    }
    config['explore'] = False
    # config['exploration_config'] = {
    #     # The Exploration class to use.
    #     'type': 'EpsilonGreedy',
    #     # Config for the Exploration class' constructor:
    #     'initial_epsilon': 1.0,
    #     'epsilon_timesteps': 400000,  # Timesteps over which to anneal epsilon.
    #     'final_epsilon': 0
    # }

    # Register env
    register_env('simple_spread', lambda config: mpe_env(simple_spread_v2))

    # Initialize ray and trainer object
    ray.init(
        num_cpus=7,
        ignore_reinit_error=True,
        # log_to_driver=False
    )

    # Stop criteria
    stop = {
        # "episode_reward_mean": -115,
        "training_iteration": 1400,
    }
    # trainer = get_trainer_class(alg_name)(env='simple_spread', config=config)
    # model = trainer.get_policy(policy_id='same').model
    # print(model.base_model.summary())
    # print(model.q_value_head.summary())
    # print(model.state_value_head.summary())
    # trainer.train()

    # Train
    results = tune.run(
        alg_name,
        stop=stop,
        config=config,
        checkpoint_at_end=True,
        checkpoint_freq=100,
        restore='/home/jiekaijia/ray_results/DQN/DQN_simple_spread_00b5b_00000_0_2021-06-02_11-59-37/checkpoint_000700/checkpoint-700',
        # resources_per_trial={"cpu": 2},
        num_samples=1
    )

    # Get the tuple of checkpoint_path and metric
    checkpoints = results.get_trial_checkpoints_paths(
        trial=results.get_best_trial("episode_reward_mean", mode="max"),
        metric="episode_reward_mean"
    )

    # # Write the checkpoint_file in the .txt file
    # with open('checkpoints.txt', 'w') as f:
    #     f.write(checkpoints[0][0])

    ray.shutdown()

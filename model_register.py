""""""

from gym.spaces import Discrete, MultiDiscrete
import numpy as np
from ray.rllib.agents.dqn import DQNTFPolicy
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.agents.dqn.dqn_tf_policy import compute_q_values, QLoss
from ray.rllib.contrib.maddpg.maddpg import MADDPGTrainer
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.tf_ops import one_hot
from ray.rllib.utils.typing import TensorType
from ray.tune.registry import register_trainable
import tensorflow as tf


PRIO_WEIGHTS = "weights"


def register_maddpg():
    class CustomStdOut(object):
        def _log_result(self, result):
            if result["training_iteration"] % 50 == 0:
                try:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        result["timesteps_total"],
                        result["episodes_total"],
                        result["episode_reward_mean"],
                        result["policy_reward_mean"],
                        round(result["time_total_s"] - self.cur_time, 3)
                    ))
                except:
                    pass

                self.cur_time = result["time_total_s"]

    MADDPGAgent = MADDPGTrainer.with_updates(
        mixins=[CustomStdOut]
    )
    register_trainable("MADDPG", MADDPGAgent)


def register_dqn():
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

    def build_q_losses(policy: Policy, model, _, train_batch: SampleBatch) -> TensorType:
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
    return MYDQNTFPolicy


def custom_model():
    class MyKerasQModel(DistributionalQTFModel):
        """Custom model for DQN."""

        def __init__(self, obs_space, action_space, num_outputs, model_config,
                     name, **kw):
            super(MyKerasQModel, self).__init__(
                obs_space, action_space, num_outputs, model_config, name, **kw)

            # Define the core model layers which will be used by the other
            # output heads of DistributionalQModel
            self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(SampleBatch.ACTIONS,
                                                                               space=self.action_space,
                                                                               shift=-1)
            self.view_requirements[SampleBatch.ACTIONS] = ViewRequirement(SampleBatch.ACTIONS, space=self.action_space,
                                                                          shift=0)
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



"""Example of using RLlib's debug callbacks.
Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.")
parser.add_argument("--stop-iters", type=int, default=2000)


class MyCallbacks(DefaultCallbacks):
    # def on_episode_start(self,
    #                      *,
    #                      worker: "RolloutWorker",
    #                      base_env: BaseEnv,
    #                      policies: Dict[PolicyID, Policy],
    #                      episode: MultiAgentEpisode,
    #                      env_index: Optional[int] = None,
    #                      **kwargs):
    #     # Make sure this episode has just been started (only initial obs
    #     # logged so far).
    #     assert episode.length == 0, \
    #         "ERROR: `on_episode_start()` callback should be called right " \
    #         "after env reset!"
    #     print("episode {} (env-idx={}) started.".format(
    #         episode.episode_id, env_index))
    #     episode.user_data["pole_angles"] = []
    #     episode.hist_data["pole_angles"] = []
    #
    # def on_episode_end(self,
    #                    *,
    #                    worker: "RolloutWorker",
    #                    base_env: BaseEnv,
    #                    policies: Dict[PolicyID, Policy],
    #                    episode: MultiAgentEpisode,
    #                    env_index: Optional[int] = None,
    #                    **kwargs):
    #     # Make sure this episode is really done.
    #     assert episode.batch_builder.policy_collectors[
    #         "default_policy"].buffers["dones"][-1], \
    #         "ERROR: `on_episode_end()` should only be called " \
    #         "after episode is done!"
    #     pole_angle = np.mean(episode.user_data["pole_angles"])
    #     print("episode {} (env-idx={}) ended with length {} and pole "
    #           "angles {}".format(episode.episode_id, env_index, episode.length,
    #                              pole_angle))
    #     episode.custom_metrics["pole_angle"] = pole_angle
    #     episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]
    #
    # def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
    #                   **kwargs):
    #     print("returned sample batch of size {}".format(samples.count))
    #
    # def on_train_result(self, *, trainer, result: dict, **kwargs):
    #     print("trainer.train() result: {} -> {} episodes".format(
    #         trainer, result["episodes_this_iter"]))
    #     # you can mutate the result dict to add new fields to return
    #     result["callback_ok"] = True
    #
    # def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
    #                       result: dict, **kwargs) -> None:
    #     result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
    #     print("policy.learn_on_batch() result: {} -> sum actions: {}".format(
    #         policy, result["sum_actions_in_train_batch"]))

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        print("postprocessed {} steps".format(postprocessed_batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1


if __name__ == "__main__":
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from pettingzoo.mpe import simple_spread_v2
    from ray.tune.registry import register_env
    args = parser.parse_args()
    def mpe_env(base_env):
        env = base_env.parallel_env()
        env = ParallelPettingZooEnv(env)
        return env


    register_env('simple_spread', lambda config: mpe_env(simple_spread_v2))

    ray.init()
    trials = tune.run(
        "DQN",
        stop={
            "training_iteration": args.stop_iters,
        },
        config={
            "env": "simple_spread",
            "num_envs_per_worker": 2,
            "callbacks": MyCallbacks,
            "framework": args.framework,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        })


    # # Verify episode-related custom metrics are there.
    # custom_metrics = trials[0].last_result["custom_metrics"]
    # print(custom_metrics)
    # assert "pole_angle_mean" in custom_metrics
    # assert "pole_angle_min" in custom_metrics
    # assert "pole_angle_max" in custom_metrics
    # assert "num_batches_mean" in custom_metrics
    # assert "callback_ok" in trials[0].last_result
    #
    # # Verify `on_learn_on_batch` custom metrics are there (per policy).
    # if args.framework == "torch":
    #     info_custom_metrics = custom_metrics["default_policy"]
    #     print(info_custom_metrics)
    #     assert "sum_actions_in_train_batch" in info_custom_metrics

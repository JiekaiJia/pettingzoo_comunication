""""""

import argparse
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
from ray.rllib.contrib.maddpg.maddpg import MADDPGTrainer
from ray.tune import run_experiments
from ray.tune.registry import register_trainable, register_env

from utils import parallel_env, parallel_comm_env, main_comm_env, main_env

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser("MADDPG with OpenAI MPE")

    # Environment
    parser.add_argument("--max-episode-len", type=int, default=25,
                        help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=25000,
                        help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0,
                        help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg",
                        help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg",
                        help="policy of adversaries")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="discount factor")
    # NOTE: 1 iteration = sample_batch_size * num_workers timesteps * num_envs_per_worker
    parser.add_argument("--sample-batch-size", type=int, default=25,
                        help="number of data points sampled /update /worker")
    parser.add_argument("--train-batch-size", type=int, default=1024,
                        help="number of data points /update")
    parser.add_argument("--n-step", type=int, default=1,
                        help="length of multistep value backup")
    parser.add_argument("--num-units", type=int, default=64,
                        help="number of units in the mlp")

    # Checkpoint
    parser.add_argument("--checkpoint-freq", type=int, default=5000,
                        help="save model once every time this many iterations are completed")
    parser.add_argument("--local-dir", type=str, default="./ray_results",
                        help="path to save checkpoints")
    parser.add_argument("--restore", type=str, default=None,
                        help="directory in which training state and model are loaded")

    # Parallelism
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-envs-per-worker", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=0)

    return parser.parse_args()


def main(args):
    ray.init()
    register_trainable("MADDPG", MADDPGTrainer)

    # Create test environment.
    env = parallel_env(simple_spread_v2)()
    # Register env
    env_name = 'simple_spread'
    register_env(env_name, lambda _: parallel_env(simple_spread_v2)())

    def gen_policy(i):
        use_local_critic = [
            args.adv_policy == "ddpg" if i < args.num_adversaries else
            args.good_policy == "ddpg" for i, _ in enumerate(env.agents)
        ]
        return (
            None,
            env.observation_space,
            env.action_space,
            {
                "agent_id": i,
                "use_local_critic": use_local_critic[i],
                "obs_space_dict": dict(zip([0, 1, 2], [env.observation_space, env.observation_space, env.observation_space])),
                "act_space_dict": dict(zip([0, 1, 2], [env.action_space, env.action_space, env.action_space])),
            }
        )

    policies = {agent: gen_policy(i) for i, agent in enumerate(env.agents)}

    ray.tune.run(
            "MADDPG",
            stop={
                "episodes_total": args.num_episodes,
            },
            checkpoint_at_end=True,
            checkpoint_freq=args.checkpoint_freq,
            local_dir=args.local_dir,
            # restore='/home/jiekaijia/PycharmProjects/pettingzoo_comunication/ray_results/MADDPG/MADDPG_mpe_2ff0a_00000_0_2021-06-07_14-23-38/checkpoint_000119/checkpoint-119',
            # args.restore,
            config={
                # === Log ===
                "log_level": "ERROR",
                'render_env': True,

                # === Environment ===
                'env': env_name,
                # "env_config": {'max_cycles': 25, 'num_agents': 3, 'local_ratio': 0.5},
                "num_envs_per_worker": args.num_envs_per_worker,
                "horizon": args.max_episode_len,

                # === Policy Config ===
                # --- Model ---
                "good_policy": args.good_policy,
                "adv_policy": args.adv_policy,
                "actor_hiddens": [args.num_units] * 2,
                "actor_hidden_activation": "relu",
                "critic_hiddens": [args.num_units] * 2,
                "critic_hidden_activation": "relu",
                "n_step": args.n_step,
                "gamma": args.gamma,

                # --- Exploration ---
                "tau": 0.01,

                # --- Replay buffer ---
                "buffer_size": int(1e6),

                # --- Optimization ---
                "actor_lr": args.lr,
                "critic_lr": args.lr,
                "learning_starts": args.train_batch_size * args.max_episode_len,
                "rollout_fragment_length": args.sample_batch_size,
                "train_batch_size": args.train_batch_size,
                "batch_mode": "truncate_episodes",

                # --- Parallelism ---
                "num_workers": args.num_workers,
                "num_gpus": int(os.environ.get('RLLIB_NUM_GPUS', '0')),
                # "num_gpus_per_worker": 0,

                # === Multi-agent setting ===
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": lambda agent_id: agent_id
                }}
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)
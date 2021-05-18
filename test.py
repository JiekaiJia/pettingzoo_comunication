""""""
from copy import deepcopy
import os


from numpy import float32
from pettingzoo.mpe import simple_spread_v2
import ray
from ray.rllib.agents.dqn import DQNTorchPolicy
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.env import PettingZooEnv
from ray import tune
from ray.tune.registry import register_env
from supersuit import dtype_v0

if __name__ == '__main__':
    """For this script, you need:
    1. Algorithm name and according module, e.g.: 'DQN' + agents.dqn as agent
    2. Name of the aec game you want to train on, e.g.: 'simple_spread'.
    3. num_cpus
    """

    # The used algorithm
    alg_name = 'DQN'
    # Environment parameters
    num_agents = 3
    local_ratio = 0.1
    max_cycles = 25

    # function that outputs the environment you wish to register.
    def env_creator(config):
        my_env = simple_spread_v2.env(
            max_cycles=config.get('max_cycles', max_cycles),
            N=config.get('num_agents', num_agents),
            local_ratio=config.get('local_ratio', local_ratio)
        )
        my_env = dtype_v0(my_env, dtype=float32)
        # env = color_reduction_v0(env, mode='R')
        # env = normalize_obs_v0(env)
        return my_env

    # Gets default training configuration.
    config = deepcopy(get_trainer_class(alg_name)._default_config)

    # Create test environment.
    test_env = PettingZooEnv(env_creator({}))

    # Extract space dimensions
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    # === Settings for Rollout Worker processes ===
    # Use GPUs if `RLLIB_NUM_GPUS` env var set to > 0.
    config['num_gpus'] = int(os.environ.get('RLLIB_NUM_GPUS', '0'))
    # Number of rollout worker actors to create for parallel sampling.
    config['num_workers'] = 3

    # === Settings for the Trainer process ===
    # Whether layers should be shared for the value function.
    config['model'] = {'vf_share_layers': True}

    # === Environment Settings ===
    config['env'] = 'simple_spread'
    # the env_creator function via the register env lambda below.
    config['env_config'] = {'max_cycles': max_cycles, 'num_agents': num_agents, 'local_ratio': local_ratio}
    # Discount factor of the MDP.
    config['gamma'] = 0.99
    # If True, try to render the environment on the local worker or on worker 1
    config['render_env'] = False
    # If True, stores videos in this relative directory inside the default
    config['record_env'] = False

    # === Debug Settings ===
    # Periodically print out summaries of relevant internal dataflow(DEBUG, INFO, WARN, or ERROR.)
    config['log_level'] = 'INFO'

    # === Deep Learning Framework Settings ===
    config['framework'] = 'torch'

    # === Settings for Multi-Agent Environments ===
    # Configuration for multi-agent setup with policy sharing:
    config['multiagent'] = {
        # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
        # of (policy_cls, obs_space, act_space, config). This defines the
        # observation and action spaces of the policies and any extra config.
        'policies': {
            f'agent_{i}': (DQNTorchPolicy, obs_space, act_space, {}) for i in range(num_agents)
        },
        # Function mapping agent ids to policy ids.
        'policy_mapping_fn': lambda agent_id: agent_id,
    }

    # DQN-specific configs
    # === Model ===
    # N-step Q learning
    config['n_step'] = 1

    # === Optimization ===
    # How many steps of the model to sample before learning starts.
    config['learning_starts'] = 1000
    # Learning rate for adam optimizer.
    config['lr'] = 0.0001
    # Divide episodes into fragments of this many steps from each worker and for each agent during rollouts!
    config['rollout_fragment_length'] = 8
    # Training batch size -> Fragments are concatenated up to this point.
    config['train_batch_size'] = 32

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    config['buffer_size'] = 50000

    # === Exploration Settings ===
    # Switch to greedy actions in evaluation workers.
    config['exploration_config'] = {
                                       # The Exploration class to use.
                                       'type': 'EpsilonGreedy',
                                       # Config for the Exploration class' constructor:
                                       'initial_epsilon': 1.0,
                                       'epsilon_timesteps': 200000,  # Timesteps over which to anneal epsilon.
                                       'final_epsilon': .01
                                  }

    # Register env
    register_env('simple_spread', lambda config: PettingZooEnv(env_creator(config)))

    # Initialize ray and trainer object
    ray.init(
        num_cpus=4,
        ignore_reinit_error=True,
        log_to_driver=False
    )

    trainer = get_trainer_class(alg_name)(env='simple_spread', config=config)

    with open('checkpoints.txt', 'r') as f:
        checkpoint_path = f.read()

    trainer.restore(checkpoint_path)
    cumulative_reward = 0
    env = simple_spread_v2.env(max_cycles=50)
    env.reset()

    for agent in env.agent_iter():
        env.render()
        observation, reward, done, info = env.last()
        if not done:
            action = trainer.compute_action(observation, policy_id=agent)
        else:
            action = None
        env.step(action)
        cumulative_reward += reward

    print(cumulative_reward)
    ray.shutdown()


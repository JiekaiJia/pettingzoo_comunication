"""This module provide two generic communication wrappers to Pettingzoo.MPE."""

import copy

from gym.spaces import Box, Discrete
import numpy as np
from pettingzoo.utils.conversions import to_parallel_wrapper
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.wrappers import AssertOutOfBoundsWrapper, OrderEnforcingWrapper
from pettingzoo.utils.wrappers.base import BaseWrapper
from scipy.special import softmax


class CommWrapper(BaseWrapper):
    """This wrapper adds communication channel to the environment, it's specific to pettingzoo.env()."""
    def __init__(self, base_env, comm_bits, **kwargs):
        raw_env = base_env.raw_env(**kwargs)
        # Set all agents to silent
        for agent in raw_env.world.agents:
            agent.silent = True
        env = AssertOutOfBoundsWrapper(raw_env)
        env = OrderEnforcingWrapper(env)
        super().__init__(env)
        self.comm_bits = comm_bits
        self.comm_method = 'Rial'
        self.n_actions = {agent: self.env.action_spaces[agent].n for agent in self.possible_agents}
        self.comm_agents = {
            agent: CommAgent(self.comm_bits, agent, self.env, self.comm_method) for agent in self.possible_agents
        }
        self.n_agents = {}
        pre_key = None
        count = 0
        for agent1 in self.possible_agents:
            key = agent1[:5]
            if key != pre_key:
                count = 0
                pre_key = key
                for agent2 in self.possible_agents:
                    if key in agent2 or key in 'listener' or 'alice' in agent2:
                        count += 1
            self.n_agents[agent1] = count

        # Extend the observation space with message.
        self.observation_spaces = {
            k: Box(low=-np.float32(np.inf), high=+np.float32(np.inf),
                   shape=(v.shape[0] + (self.n_agents[k] - 1) * comm_bits,), dtype=np.float32)
            if isinstance(v, Box) else Discrete(v.n + (self.n_agents[k] - 1) * comm_bits) for k, v in
            env.observation_spaces.items()
        }

        factor = {}
        for k, v in self.comm_agents.items():
            if v.receivers:
                # If agent has receivers, then it will have comm_action
                factor[k] = 1
            else:
                # If agent doesn't have receivers, then it will have no comm_action
                factor[k] = 0
        # Extend the action space with communication actions.
        self.action_spaces = {
            k: Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(v.shape[0] * comm_bits ** factor[k],),
                   dtype=np.float32)
            if isinstance(v, Box) else Discrete(v.n * comm_bits ** factor[k]) for k, v in env.action_spaces.items()
        }

    def last(self, observe=True):
        msg = []
        observation, reward, done, info = super().last(observe)
        self.receive_message()
        for message in self.comm_agents[self.agent_selection].input_queue:
            msg.append(message.message)
        msg = np.reshape(msg, (-1))
        obs = np.concatenate((observation, msg))
        # Make sure the obs has the same size with observation_space.
        obs = np.concatenate((obs, np.zeros(self.observation_spaces[self.agent_selection].shape[0] - len(obs))))

        return obs, reward, done, info

    def step(self, action):
        # Compute physical actions and communication actions.
        if isinstance(action, (np.int64, np.int32, np.int16, np.int8, int)):
            act = action % self.n_actions[self.agent_selection]
            self.comm_agents[self.agent_selection].comm_action = action // self.n_actions[self.agent_selection]
        elif not action:
            act = action
        else:
            number = np.argmax(action)
            act = number % self.n_actions[self.agent_selection]
            self.comm_agents[self.agent_selection].comm_action = number // self.n_actions[self.agent_selection]
        self.update_messages()
        self.send_messages()
        super().step(act)

    def update_messages(self):
        self.comm_agents[self.agent_selection].message_update()

    def send_messages(self):
        self.comm_agents[self.agent_selection].send_messages()

    def receive_message(self):
        self.comm_agents[self.agent_selection].receive_messages(self.comm_agents)

    def __str__(self):
        return str(self.env)


class ParallelCommWrapper(ParallelEnv):
    """This wrapper adds communication channel to the environment, it's specific to pettingzoo.parallelenv()."""

    def __init__(self, base_env, comm_bits, **kwargs):
        raw_env = base_env.raw_env(**kwargs)

        # Set all agents to silent
        for agent in raw_env.world.agents:
            agent.silent = True
        env = AssertOutOfBoundsWrapper(raw_env)
        env = OrderEnforcingWrapper(env)
        env = to_parallel_wrapper(env)
        self.env = env
        self.possible_agents = env.possible_agents
        self.comm_bits = comm_bits
        self.comm_method = 'Rial'
        self.n_actions = {agent: self.env.action_spaces[agent].n for agent in self.possible_agents}
        self.comm_agents = {
            agent: CommAgent(self.comm_bits, agent, self.env, self.comm_method) for agent in self.possible_agents
        }
        self.n_agents = {}
        pre_key = None
        count = 0
        for agent1 in self.possible_agents:
            key = agent1[:5]
            if key != pre_key:
                count = 0
                pre_key = key
                for agent2 in self.possible_agents:
                    if key in agent2 or key in 'listener' or 'alice' in agent2:
                        count += 1
            self.n_agents[agent1] = count
        # Extend the observation space with message.
        self.observation_spaces = {
            k: Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(v.shape[0] + (self.n_agents[k] - 1) * comm_bits,), dtype=np.float32)
            if isinstance(v, Box) else Discrete(v.n + (self.n_agents[k] - 1) * comm_bits) for k, v in env.observation_spaces.items()
        }

        factor = {}
        for k, v in self.comm_agents.items():
            if v.receivers:
                # If agent has receivers, then it will have comm_action
                factor[k] = 1
            else:
                # If agent doesn't have receivers, then it will have no comm_action
                factor[k] = 0
        # Extend the action space with communication actions.
        self.action_spaces = {
            k: Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(v.shape[0]*comm_bits**factor[k],), dtype=np.float32)
            if isinstance(v, Box) else Discrete(v.n*comm_bits**factor[k]) for k, v in env.action_spaces.items()
        }

        self.metadata = env.metadata

        # Not every environment has the .state_space attribute implemented
        try:
            self.state_space = self.env.state_space
        except AttributeError:
            pass

    def unwrapped(self):
        return self.env.unwrapped()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        observations = self.env.reset()
        self.agents = self.env.possible_agents
        # Make sure the obs has the same size with observation_space.
        return {k: np.concatenate((v, np.zeros((self.n_agents[k] - 1) * self.comm_bits)))for k, v in observations.items()}

    def step(self, actions):
        acts = {}
        # Compute physical actions and communication actions.
        for agent in self.agents:
            if isinstance(actions[agent], (np.int64, np.int32, np.int16, np.int8, int)):
                acts[agent] = actions[agent] % self.n_actions[agent]
                self.comm_agents[agent].comm_action = actions[agent] // self.n_actions[agent]
            else:
                number = np.argmax(actions[agent])
                acts[agent] = number % self.n_actions[agent]
                self.comm_agents[agent].comm_action = number // self.n_actions[agent]
        self.update_messages()
        self.send_messages()
        self.receive_message()
        observations, rewards, dones, infos = self.env.step(acts)
        obs = {}
        for k, v in observations.items():
            msg = []
            for message in self.comm_agents[k].input_queue:
                msg.append(message.message)
            msg = np.reshape(msg, (-1))
            obs[k] = np.concatenate((v, msg))

        return obs, rewards, dones, infos

    def render(self, mode="human"):
        return self.env.render(mode)

    def state(self):
        return self.env.state()

    def close(self):
        return self.env.close()

    def update_messages(self):
        for agent in self.agents:
            self.comm_agents[agent].message_update()

    def send_messages(self):
        for agent in self.agents:
            self.comm_agents[agent].send_messages()

    def receive_message(self):
        for agent in self.agents:
            self.comm_agents[agent].receive_messages(self.comm_agents)

    def __str__(self):
        return str(self.env)


class CommAgent:
    def __init__(self, comm_bits, agent, env, comm):
        self.name = agent
        self.env = env
        self.comm_bits = comm_bits
        self.receivers = []
        self.output_queue = []
        self.input_queue = []
        self.comm_method = comm
        self.comm_action = np.zeros(comm_bits)
        self.comm_vector = np.zeros(comm_bits)
        self.comm_state = np.zeros(comm_bits)
        self.reset()

    def reset(self):
        self._pick_receiver()
        # print(self.name)
        # print(self.receivers)
        self.messages = [Message(self.name, receiver, self.comm_state) for receiver in self.receivers]

    def _pick_receiver(self):
        for agent in self.env.possible_agents:
            # Make sure agent not to receive itself message
            if agent is not self.name:
                # Suppose, only agents with same type can exchange message
                if agent[:5] in self.name or agent[:5] in 'listener' or 'alice' in self.name:
                    self.receivers.append(agent)

    def message_update(self):
        # noise = np.random.randn(self.comm_bits)
        noise = 0
        if self.comm_method == 'Rial':
            onehot = np.zeros(self.comm_bits)
            onehot[self.comm_action] = 1
            self.comm_state = softmax(onehot + noise)
        elif self.comm_method == 'Dial':
            self.comm_state = softmax(self.comm_vector + noise)

        for message in self.messages:
            message.update(self.comm_state)

    def send_messages(self):
        # Put message in the output queue
        self.output_queue = []
        self.output_queue.extend(copy.deepcopy(self.messages))
        for message in self.messages:
            # print(f'Agent {self.name} sent message: {message.to_json()}')
            pass

    def receive_messages(self, comm_agents):
        self.input_queue = []
        for _, agent in comm_agents.items():
            if agent.name is not self.name and self.name in agent.receivers:
                # Save feedback and do something with it.
                for message in agent.output_queue:
                    if message.receiver is self.name:
                        self.input_queue.append(message)
                        agent.output_queue.remove(message)
                        # print(f'{self.name} received message: {message.to_json()}')


class Message:
    def __init__(self, sender, receiver, message):
        self.sender = sender
        self.receiver = receiver
        self.message = message

    def update(self, message):
        self.message = message

    def to_json(self):
        return f'senderID: {self.sender}, receiverID: {self.receiver}, message: {self.message}'


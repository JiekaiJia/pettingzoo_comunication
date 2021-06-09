"""This module provide two generic communication wrappers to Pettingzoo."""

import copy

from gym.spaces import Box, Discrete
import numpy as np
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.wrappers.base import BaseWrapper
from scipy.special import softmax


class CommWrapper(BaseWrapper):
    """This wrapper adds communication channel to the environment, it's specific to pettingzoo.env().
    env:
    receivers: A dictionary that includes information about communication.
    Example: receivers = {
                 agent_0: [agent_1, agent_2],
                 agent_1: [agent_0, agent_2],
                 agent_2: [agent_0, agent_1],
                 adversary_0: [adversary_1],
                 adversary_1: [adversary_0],
            }"""
    def __init__(self, env, comm_dict):
        super().__init__(env)
        self.comm_bits = comm_dict['comm_bits']
        self.comm_method = 'RIAL'
        receivers = comm_dict['receivers']
        self.comm_agents = {
            agent: CommAgent({'comm_bits': self.comm_bits, 'receivers': receivers[agent]}, agent, self.env, self.comm_method) for agent in self.possible_agents
        }
        self.n_messages = {}
        for agent in self.possible_agents:
            count = 0
            for _, rec_list in receivers.items():
                if agent in rec_list:
                    count += 1
            self.n_messages[agent] = count

        # Extend the observation space with message.
        self.obs_mapping_size = {}
        self.observation_spaces = copy.deepcopy(env.observation_spaces)
        for k, v in self.observation_spaces.items():
            if isinstance(v, Box):
                if len(v.shape) == 1:
                    # Add message to 1-D obs
                    v.shape = (v.shape[0] + self.n_messages[k] * self.comm_bits,)
                else:
                    # Add message to 3-D obs, the message will be taken as 4th channel. If one channel can't hold
                    # all the messages, then add another channel. The obs is the channel last type.
                    self.obs_mapping_size[k] = v.shape[0]*v.shape[1]
                    comm_channel = (self.n_messages[k] * self.comm_bits) // (v.shape[0]*v.shape[1])
                    comm_rest = (self.n_messages[k] * self.comm_bits) % (v.shape[0]*v.shape[1])
                    if comm_rest == 0:
                        pass
                    else:
                        comm_channel += 1
                    tmp = list(v.shape)
                    tmp[-1] += comm_channel
                    v.shape = tuple(tmp)
            elif isinstance(v, Discrete):
                v.n = v.n + self.n_messages[k] * self.comm_bits
        # Compute the actions spaces before adding comm_channel
        if isinstance(env.action_spaces[self.possible_agents[0]], Discrete):
            self.n_actions = {agent: env.action_spaces[agent].n for agent in self.possible_agents}
        elif isinstance(env.action_spaces[self.possible_agents[0]], Box):
            self.n_actions = {agent: env.action_spaces[agent].shape[0] for agent in self.possible_agents}

        if self.comm_method == 'RIAL':
            self.factor = {}
            for k, v in self.comm_agents.items():
                if v.receivers:
                    # If agent has receivers, then it will have comm_action
                    self.factor[k] = 1
                else:
                    # If agent doesn't have receivers, then it will have no comm_action
                    self.factor[k] = 0
            # Extend the action space with communication actions.
            self.action_spaces = copy.deepcopy(env.action_spaces)
            for k, v in self.action_spaces.items():
                if isinstance(v, Box):
                    v.shape = (v.shape[0] * self.comm_bits ** self.factor[k],)
                elif isinstance(v, Discrete):
                    v.n = v.n * self.comm_bits ** self.factor[k]
        elif self.comm_method == 'DIAL':
            self.action_spaces = env.action_spaces

    def last(self, observe=True):
        msg = []
        observation, reward, done, info = super().last(observe)
        self._receive_message()
        for message in self.comm_agents[self.agent_selection].input_queue:
            msg.append(message.message)
        msg = np.reshape(msg, (-1))
        if len(observation.shape) == 1:
            obs = np.concatenate((observation, msg))
            # Make sure the obs has the same size with observation_space.
            obs = np.concatenate((obs, np.zeros(self.observation_spaces[self.agent_selection].shape[0] - len(obs))))
        else:
            rest_n = self.observation_spaces[self.agent_selection].shape[-1] - observation.shape[-1]
            if rest_n == 0:
                obs = observation
            else:
                msg = np.pad(msg, (0, rest_n*self.obs_mapping_size[self.agent_selection]), 'constant', constant_values=(0, 0))
                msg = msg.reshape(rest_n, self.observation_spaces[self.agent_selection].shape)
                obs = np.concatenate((observation, msg), axis=-1)
        return obs, reward, done, info

    def step(self, action, comm_vector=None):
        # Compute physical actions and communication actions.
        self.comm_agents[self.agent_selection].comm_vector = comm_vector
        if action is None:
            act = None
        elif isinstance(self.action_spaces[self.agent_selection], Discrete):
            if isinstance(action, (np.int64, np.int32, np.int16, np.int8, int)):
                act = action % self.n_actions[self.agent_selection]
                self.comm_agents[self.agent_selection].comm_action = action // self.n_actions[self.agent_selection]
            else:
                number = np.argmax(action)
                act = number % self.n_actions[self.agent_selection]
                self.comm_agents[self.agent_selection].comm_action = number // self.n_actions[self.agent_selection]
        elif isinstance(self.action_spaces[self.agent_selection], Box):
            acts = action.reshape(self.comm_bits**self.factor[self.agent_selection], self.n_actions[self.agent_selection])
            act_norm = [np.linalg.norm(acts[x, :]) for x in range(self.comm_bits**self.factor[self.agent_selection])]
            idx = np.argmax(act_norm)
            self.comm_agents[self.agent_selection].comm_action = idx
            act = acts[idx, :].reshape(self.n_actions[self.agent_selection],)

        self._update_messages()
        self._send_messages()
        super().step(act)

    def _update_messages(self):
        self.comm_agents[self.agent_selection].message_update()

    def _send_messages(self):
        self.comm_agents[self.agent_selection].send_messages()

    def _receive_message(self):
        self.comm_agents[self.agent_selection].receive_messages(self.comm_agents)

    def __str__(self):
        return str(self.env)


class ParallelCommWrapper(ParallelEnv):
    """This wrapper adds communication channel to the environment, it's specific to pettingzoo.parallelenv().
    env:
    receivers: A dictionary that includes information about communication.
    Example: receivers = {
                 agent_0: [agent_1, agent_2],
                 agent_1: [agent_0, agent_2],
                 agent_2: [agent_0, agent_1],
                 adversary_0: [adversary_1],
                 adversary_1: [adversary_0],
            }"""
    def __init__(self, env, comm_dict):
        self.env = env
        self.comm_bits = comm_dict['comm_bits']
        self.possible_agents = env.possible_agents
        self.comm_method = 'RIAL'
        receivers = comm_dict['receivers']
        self.comm_agents = {
            agent: CommAgent({'comm_bits': self.comm_bits, 'receivers': receivers[agent]}, agent, self.env, self.comm_method) for agent in self.possible_agents
        }
        self.n_messages = {}
        for agent in self.possible_agents:
            count = 0
            for _, rec_list in receivers.items():
                if agent in rec_list:
                    count += 1
            self.n_messages[agent] = count

        # Extend the observation space with message.
        self.observation_spaces = copy.deepcopy(env.observation_spaces)
        for k, v in self.observation_spaces.items():
            if isinstance(v, Box):
                if len(v.shape) == 1:
                    # Add message to 1-D obs.
                    v.shape = (v.shape[0] + self.n_messages[k] * self.comm_bits,)
                else:
                    # Add message to 3-D obs, the message will be taken as 4th channel. If one channel can't hold
                    # all the messages, then add another channel. The obs is the channel last type.
                    comm_channel = (self.n_messages[k] * self.comm_bits) // (v.shape[0]*v.shape[1])
                    comm_rest = (self.n_messages[k] * self.comm_bits) % (v.shape[0]*v.shape[1])
                    if comm_rest == 0:
                        pass
                    else:
                        comm_channel += 1
                    tmp = list(v.shape)
                    tmp[-1] += comm_channel
                    v.shape = tuple(tmp)
            elif isinstance(v, Discrete):
                v.n = v.n + self.n_messages[k] * self.comm_bits
        # Compute the actions spaces before adding comm_channel
        if isinstance(env.action_spaces[self.possible_agents[0]], Discrete):
            self.n_actions = {agent: env.action_spaces[agent].n for agent in self.possible_agents}
        elif isinstance(env.action_spaces[self.possible_agents[0]], Box):
            self.n_actions = {agent: env.action_spaces[agent].shape[0] for agent in self.possible_agents}

        if self.comm_method == 'RIAL':
            self.factor = {}
            for k, v in self.comm_agents.items():
                if v.receivers:
                    # If agent has receivers, then it will have comm_action
                    self.factor[k] = 1
                else:
                    # If agent doesn't have receivers, then it will have no comm_action
                    self.factor[k] = 0
            # Extend the action space with communication actions.
            self.action_spaces = copy.deepcopy(env.action_spaces)
            for k, v in self.action_spaces.items():
                if isinstance(v, Box):
                    v.shape = (v.shape[0] * self.comm_bits ** self.factor[k],)
                elif isinstance(v, Discrete):
                    v.n = v.n * self.comm_bits ** self.factor[k]
        elif self.comm_method == 'DIAL':
            self.action_spaces = env.action_spaces

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
        return {k: np.concatenate((v, np.zeros(self.n_messages[k] * self.comm_bits)))for k, v in observations.items()}

    def step(self, actions, comm_vectors=None):
        acts = {}
        if comm_vectors:
            for agent in self.agents:
                self.comm_agents[agent].comm_vector = comm_vectors[agent]
        # Compute physical actions and communication actions.
        if isinstance(self.action_spaces[self.agents[0]], Discrete):
            for agent in self.agents:
                if isinstance(actions[agent], (np.int64, np.int32, np.int16, np.int8, int)):
                    acts[agent] = actions[agent] % self.n_actions[agent]
                    self.comm_agents[agent].comm_action = actions[agent] // self.n_actions[agent]
                else:
                    number = np.argmax(actions[agent])
                    acts[agent] = number % self.n_actions[agent]
                    self.comm_agents[agent].comm_action = number // self.n_actions[agent]
        elif isinstance(self.action_spaces[self.agents[0]], Box):
            for agent in self.agents:
                act = actions[agent].reshape(self.comm_bits ** self.factor[agent], self.n_actions[agent])
                act_norm = [np.linalg.norm(act[x, :]) for x in range(self.comm_bits ** self.factor[agent])]
                idx = np.argmax(act_norm)
                self.comm_agents[agent].comm_action = idx
                acts[agent] = act[idx, :].reshape(self.n_actions[agent],)

        self._update_messages()
        self._send_messages()
        self._receive_message()
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

    def _update_messages(self):
        for agent in self.agents:
            self.comm_agents[agent].message_update()

    def _send_messages(self):
        for agent in self.agents:
            self.comm_agents[agent].send_messages()

    def _receive_message(self):
        for agent in self.agents:
            self.comm_agents[agent].receive_messages(self.comm_agents)

    def __str__(self):
        return str(self.env)


class CommAgent:
    def __init__(self, comm_dict, agent, env, comm):
        self.name = agent
        self.env = env
        self.comm_bits = comm_dict['comm_bits']
        self.receivers = comm_dict['receivers']
        self.output_queue = []
        self.input_queue = []
        self.comm_method = comm
        self.comm_action = np.zeros(self.comm_bits)
        self.comm_vector = np.zeros(self.comm_bits)
        self.comm_state = np.zeros(self.comm_bits)
        self.messages = [Message(self.name, receiver, self.comm_state) for receiver in self.receivers]

    def message_update(self):
        # noise = np.random.randn(self.comm_bits)
        noise = 0
        if self.comm_method == 'RIAL':
            onehot = np.zeros(self.comm_bits)
            onehot[self.comm_action] = 1
            self.comm_state = softmax(onehot + noise)
        elif self.comm_method == 'DIAL':
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


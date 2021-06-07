"""This module provide a generic communication wrapper to pettingzoo.mpe."""

import copy

import numpy as np
from pettingzoo.utils.wrappers.base import BaseWrapper
from gym.spaces import Box, Discrete
from scipy.special import softmax
import torch
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.conversions import to_parallel_wrapper
from pettingzoo.utils.wrappers import AssertOutOfBoundsWrapper, OrderEnforcingWrapper
from supersuit.action_transforms.homogenize_ops import homogenize_spaces


# class CommunicationWrapper(BaseWrapper):
#     """this wrapper adds communication channel to the environment."""
#     def __init__(self, base_env, comm_bits, parallel=True):
#         raw_env = base_env.raw_env()
#         for agent in raw_env.world.agents:
#             agent.silent = True
#         env = AssertOutOfBoundsWrapper(raw_env)
#         env = OrderEnforcingWrapper(env)
#         if parallel:
#             env = to_parallel_wrapper(env)
#         super().__init__(env)
#         self.comm_bits = comm_bits
#         self.steps = 0
#
#     def reset(self):
#         super().reset()
#         self.comm_agents = {agent: CommAgent(self.comm_bits, agent, self.env) for agent in self.agents}
#
#     def step(self, action):
#         if isinstance(action, torch.Tensor):
#             action = action.numpy()[0]
#         if isinstance(action, int):
#             pass
#         else:
#             action = np.argmax(action)
#         super().step(action)
#         self.steps += 1
#
#     def message_update(self, comm_action=None, q_comm=None):
#         self.comm_agents[self.agent_selection].message_update(comm_action=comm_action, q_comm=q_comm)
#
#     def get_c_action(self):
#         return self.comm_agents[self.agent_selection].get_c_action()
#
#     def send_message(self):
#         send_messages = []
#         if (self.steps + 1) % 3 == 0:
#             for agent in self.agents:
#                 send_messages.append(self.comm_agents[agent].send_messages())
#         return send_messages
#
#     def receive_message(self):
#         return self.comm_agents[self.agent_selection].receive_messages(self.comm_agents)
#
#     # sampleMessage = Message(self.agentID, randomAgentID, str(randint(0, 1000)))
#     # self.sendMessage(sampleMessage)
#
#     def __str__(self):
#         return str(self.env)


class ParallelCommWrapper(ParallelEnv):
    """This wrapper adds communication channel to the environment."""

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
                    if key in agent2 or key in 'listener':
                        count += 1
            self.n_agents[agent1] = count

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

        self.action_spaces = {
            k: Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(v.shape[0]*comm_bits**factor[k],), dtype=np.float32)
            if isinstance(v, Box) else Discrete(v.n*comm_bits**factor[k]) for k, v in env.action_spaces.items()
        }
        self.all_n_actions = {k: v.n for k, v in self.action_spaces.items()}

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
        return observations

    def step(self, actions):
        acts = {}
        for agent in self.agents:
            if isinstance(actions[agent], (np.int64, np.int32, np.int16, np.int8, int)):
                # scaling_act = int(actions[agent] / self.action_spaces[agent].n * self.all_n_actions[agent])
                acts[agent] = actions[agent] % self.n_actions[agent]
                self.comm_agents[agent].comm_action = actions[agent] // self.n_actions[agent]
            else:
                number = np.argmax(actions[agent])
                # scaling_act = int(number / self.action_spaces[agent].n * self.all_n_actions[agent])
                acts[agent] = number % self.n_actions[agent]
                self.comm_agents[agent].comm_action = number // self.n_actions[agent]
        self.update_messages()
        self.send_messages()
        self.receive_message()
        observations, rewards, dones, infos = self.env.step(acts)
        obs = {}
        for k, v in observations.items():
            # print(v.shape)
            # print(self.observation_spaces[k].shape)

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
                if agent[:5] in self.name or agent[:5] in 'listener':
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


import copy

import numpy as np
from pettingzoo.utils.wrappers.base import BaseWrapper
from gym.spaces import Box, Discrete
from scipy.special import softmax
import torch

from dru import DRU


class CommunicationWrapper(BaseWrapper):
    """this wrapper adds communication channel to the environment."""
    def __init__(self, env, comm_bits):
        super().__init__(env)
        self.comm_bits = comm_bits
        self.steps = 0

    def reset(self):
        super().reset()
        self.comm_agents = {agent: CommAgent(self.comm_bits, agent, self.env) for agent in self.agents}

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.numpy()[0]
        if isinstance(action, int):
            pass
        else:
            action = np.argmax(action)
        super().step(action)
        self.steps += 1

    def message_update(self, comm_action=None, q_comm=None):
        self.comm_agents[self.agent_selection].message_update(comm_action=comm_action, q_comm=q_comm)

    def get_c_action(self):
        return self.comm_agents[self.agent_selection].get_c_action()

    def send_message(self):
        send_messages = []
        if (self.steps + 1) % 3 == 0:
            for agent in self.agents:
                send_messages.append(self.comm_agents[agent].send_messages())
        return send_messages

    def receive_message(self):
        return self.comm_agents[self.agent_selection].receive_messages(self.comm_agents)

    # sampleMessage = Message(self.agentID, randomAgentID, str(randint(0, 1000)))
    # self.sendMessage(sampleMessage)

    def __str__(self):
        return str(self.env)


### not finished
class ParallelCommWrapper:
    """this wrapper adds communication channel to the environment."""
    def __init__(self, env, comm_bits):
        self.env = env
        self.possible_agents = env.possible_agents
        self.nagents = len(self.possible_agents)
        self.observation_spaces = {
            k: Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(v.shape[0]+(self.nagents-1)*comm_bits,), dtype=np.float32)
            if isinstance(v, Box) else Discrete(v.n+comm_bits) for k, v in env.observation_spaces.items()
        }
        self.action_spaces = {
            k: Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(v.shape[0]+comm_bits,), dtype=np.float32)
            if isinstance(v, Box) else Discrete(v.n+comm_bits) for k, v in env.action_spaces.items()
        }
        self.metadata = env.metadata
        self.comm_bits = comm_bits
        self.comm_method = 'Rial'

        # Not every environment has the .state_space attribute implemented
        try:
            self.state_space = self.env.state_space
        except AttributeError:
            pass

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        observations = self.env.reset()
        tmp = np.zeros((self.nagents - 1) * self.comm_bits)
        self.agents = self.env.agents
        self.comm_agents = {agent: CommAgent(self.comm_bits, agent, self.env, self.comm_method) for agent in self.agents}
        return {k: np.concatenate((v, tmp)) for k, v in observations.items()}

    def step(self, actions):
        acts = {}
        for agent in self.agents:
            if isinstance(actions[agent], int):
                pass
            else:
                acts[agent] = np.argmax(actions[agent][:-self.comm_bits])
                self.comm_agents[agent].comm_action = actions[agent][-self.comm_bits:]
        self.update_messages()
        self.send_messages()
        self.receive_message()
        observations, rewards, dones, infos = self.env.step(acts)
        obs = {}
        for k, v in observations.items():
            msg = []
            print(len(self.comm_agents[k].input_queue))
            for message in self.comm_agents[k].input_queue:
                msg.append(message.message)

            obs[k] = np.concatenate((v, np.reshape(msg, (-1))))

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

    # sampleMessage = Message(self.agentID, randomAgentID, str(randint(0, 1000)))
    # self.sendMessage(sampleMessage)

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

    def reset(self, limited=False):
        self._pick_receiver(limited=limited)
        self.messages = [Message(self.name, receiver, self.comm_state) for receiver in self.receivers]

    def _pick_receiver(self, limited=False):
        if limited:
            idx = self.env.agents.index(self.name)
            self.receivers.append(self.env.agents[idx - 1])
        else:
            for agent in self.env.agents:
                if agent is not self.name:
                    self.receivers.append(agent)

    def message_update(self):
        # noise = np.random.randn(self.comm_bits)
        noise = 0
        if self.comm_method == 'Rial':
            onehot = np.zeros(self.comm_bits)
            d = np.argmax(self.comm_action)
            onehot[d] = 1
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


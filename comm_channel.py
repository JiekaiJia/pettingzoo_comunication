import copy

import numpy as np
from pettingzoo.utils.wrappers.base import BaseWrapper
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
        self.steps += 1
        super().step(action)

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


class CommAgent:
    def __init__(self, comm_bits, agent, env):
        self.name = agent
        self.env = env
        self.comm_bits = comm_bits
        self.receivers = []
        self.comm_action = None
        self.comm_state = np.zeros(comm_bits)
        self.output_queue = []
        self.reset()

    def reset(self, limited=False):
        self._pick_receiver(limited=limited)
        self.messages = [Message(self.name, receiver, self.comm_state) for receiver in self.receivers]
        self.output_queue.extend(copy.deepcopy(self.messages))

    def _pick_receiver(self, limited=False):
        if limited:
            idx = self.env.agents.index(self.name)
            self.receivers.append(self.env.agents[idx - 1])
        else:
            for agent in self.env.agents:
                if agent is not self.name:
                    self.receivers.append(agent)

    def message_update(self, q_comm=None):
        # if isinstance(q_comm, torch.Tensor):
        #     q_comm = q_comm.detach().numpy()[0]
        if isinstance(q_comm, torch.Tensor):
            q_comm = q_comm.detach().numpy()

        # noise = np.random.randn(self.comm_bits)
        noise = 0
        if comm_action is None and q_comm is None:
            pass
        elif comm_action == 0:
            self.comm_state = q_comm + noise
        else:
            self.comm_action = comm_action
            comm_vector = np.zeros(self.comm_bits)
            comm_vector[comm_action] = 1
            self.comm_state = comm_vector + noise

        for message in self.messages:
            message.update(self.comm_state)

    def get_c_action(self):
        return self.comm_action

    def send_messages(self):
        # Put message in the output queue
        self.output_queue.extend(copy.deepcopy(self.messages))
        for message in self.messages:
            # print(f'Agent {self.name} sent message: {message.to_json()}')
            pass

        return self.output_queue

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
        return self.input_queue


class Message:
    def __init__(self, sender, receiver, message):
        self.sender = sender
        self.receiver = receiver
        self.message = message

    def update(self, message):
        self.message = message

    def to_json(self):
        return f'senderID: {self.sender}, receiverID: {self.receiver}, message: {self.message}'


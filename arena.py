import copy

import numpy as np
import torch
from torch.autograd import Variable

from dotdic import DotDic


class Arena:
	def __init__(self, opt, game):
		self.opt = opt
		self.game = game
		self.eps = opt.eps

	def create_episode(self):
		opt = self.opt
		episode = DotDic({})
		episode.r = torch.zeros(opt.bs, opt.game_nagents).float()
		episode.steps = torch.zeros(opt.bs).int()
		episode.ended = torch.zeros(opt.bs).int()
		episode.step_records = []

		return episode

	def create_step_record(self):
		opt = self.opt
		record = DotDic({})
		record.s_t = torch.zeros(opt.bs, opt.game_nagents, 18)
		record.r_t = torch.zeros(opt.bs, opt.game_nagents)
		record.terminal = torch.zeros(opt.bs)

		record.agent_inputs = []

		# Track actions at time t per agent
		record.a_t = torch.zeros(opt.bs, opt.game_nagents, dtype=torch.long)
		if not opt.model_dial:
			record.a_comm_t = torch.zeros(opt.bs, opt.game_nagents, dtype=torch.long)

		# Track messages sent at time t per agent
		if opt.comm_enabled:
			comm_dtype = opt.model_dial and torch.float or torch.long
			comm_dtype = torch.float
			record.comm = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits, dtype=comm_dtype)
			if opt.model_dial and opt.model_target:
				record.comm_target = record.comm.clone()

		# Track hidden state per time t per agent
		record.hidden = torch.zeros(opt.game_nagents, opt.model_rnn_layers, opt.bs, opt.model_rnn_size)
		record.hidden_target = torch.zeros(opt.game_nagents, opt.model_rnn_layers, opt.bs, opt.model_rnn_size)

		# Track Q(a_t) and Q(a_max_t) per agent
		record.q_a_t = torch.zeros(opt.bs, opt.game_nagents)
		record.q_a_max_t = torch.zeros(opt.bs, opt.game_nagents)

		# Track Q(m_t) and Q(m_max_t) per agent
		if not opt.model_dial:
			record.q_comm_t = torch.zeros(opt.bs, opt.game_nagents)
			record.q_comm_max_t = torch.zeros(opt.bs, opt.game_nagents)

		return record

	def run_episode(self, agents, train_mode=False):
		# TODO: implemented parallel environment to get batch
		opt = self.opt
		game = self.game
		game.reset()
		self.eps = self.eps * opt.eps_decay

		episode = self.create_episode()
		s_t = game.get_state()
		episode.step_records.append(self.create_step_record())
		episode.step_records[-1].s_t[:, :] = torch.tensor(s_t)
		step = 0
		_step = 0
		for i in self.game.env.agent_iter():
			episode.step_records.append(self.create_step_record())
			# Get received messages per agent per batch
			agent_idx = self.game.env.agents.index(i)
			agent = agents[agent_idx]
			comm = None
			if opt.comm_enabled:
				# TODO: set limited message
				comm = episode.step_records[step].comm.clone()
				comm[:, agent_idx].zero_()
				comm = torch.mean(comm, dim=1)
				comm = comm.view(-1, 1, self.opt.game_comm_bits)

			# Get prev action per batch
			prev_action = None
			if opt.model_action_aware:
				prev_action = torch.ones(opt.bs, dtype=torch.long)
				if not opt.model_dial:
					prev_message = torch.ones(opt.bs, dtype=torch.long)
				for b in range(opt.bs):
					if step > 0 and episode.step_records[step - 1].a_t[b, agent_idx] > 0:
						prev_action[b] = episode.step_records[step - 1].a_t[b, agent_idx]
					if not opt.model_dial:
						if step > 0 and episode.step_records[step - 1].a_comm_t[b, agent_idx] > 0:
							prev_message[b] = episode.step_records[step - 1].a_comm_t[b, agent_idx]
				if not opt.model_dial:
					prev_action = (prev_action, prev_message)

			# Batch agent index for input into model
			batch_agent_index = torch.zeros(opt.bs, dtype=torch.long).fill_(agent_idx)

			agent_inputs = {
				's_t': episode.step_records[step].s_t[:, agent_idx],
				'messages': comm,
				'hidden': episode.step_records[step].hidden[agent_idx, :],  # Hidden state
				'prev_action': prev_action,
				'agent_index': batch_agent_index
			}
			episode.step_records[step].agent_inputs.append(agent_inputs)

			# Compute model output (Q function + message bits)
			hidden_t, q_t = agent.model(**agent_inputs)
			q_t = q_t.view(-1, self.opt.game_action_space + self.opt.game_comm_bits)
			episode.step_records[step + 1].hidden[agent_idx] = hidden_t

			# Choose next action and comm using eps-greedy selector
			(action, action_value), (comm_vector, comm_action, comm_value) = \
				agent.select_action_and_comm(step, q_t, eps=self.eps, train_mode=train_mode)

			# Store action + comm
			episode.step_records[step].a_t[:, agent_idx] = action
			episode.step_records[step].q_a_t[:, agent_idx] = action_value
			episode.step_records[step + 1].comm[:, agent_idx] = comm_vector
			if not opt.model_dial:
				episode.step_records[step].a_comm_t[:, agent_idx] = comm_action
				episode.step_records[step].q_comm_t[:, agent_idx] = comm_value

			episode.step_records[step].r_t[:, agent_idx], done = self.game.get_reward()

			if not done:
				_action = action.numpy()[0]
			else:
				_action = None
				
			if (_step+1) % 3 == 0:
				# Accumulate steps
				# Accumulate steps
				if step < opt.nsteps:
					for b in range(opt.bs):
						if not done:
							episode.steps[b] = episode.steps[b] + 1
							episode.r[b] = episode.step_records[step].r_t[b]

				# Target-network forward pass
				if opt.model_target and train_mode:
					agent_inputs = episode.step_records[step].agent_inputs[agent_idx]
					# import pdb; pdb.set_trace()
					comm_target = agent_inputs.get('messages', None)

					if opt.comm_enabled and opt.model_dial:
						# TODO: set limited message
						all_comm = episode.step_records[step].comm_target.clone()
						comm_target = all_comm[:, torch.arange(all_comm.size(1)) != agent_idx]
						comm_target = comm_target.view([-1, (opt.game_nagents - 1) * opt.game_comm_bits])
						comm_target[:, agent_idx].zero_()

					# comm_target.retain_grad()
					agent_target_inputs = copy.copy(agent_inputs)
					agent_target_inputs['messages'] = Variable(comm_target)
					agent_target_inputs['hidden'] = episode.step_records[step].hidden_target[agent_idx, :]
					hidden_target_t, q_target_t = agent.model_target(**agent_target_inputs)
					q_target_t = q_target_t.view(-1, self.opt.game_action_space + self.opt.game_comm_bits)
					episode.step_records[step + 1].hidden_target[agent_idx] = hidden_target_t

					# Choose next arg max action and comm
					(action, action_value), (comm_vector, comm_action, comm_value) = \
						agent.select_action_and_comm(step, q_target_t, eps=0, target=True, train_mode=True)

					# save target actions, comm, and q_a_t, q_a_max_t
					episode.step_records[step].q_a_max_t[:, agent_idx] = action_value
					if opt.model_dial:
						episode.step_records[step + 1].comm_target[:, agent_idx] = comm_vector
					else:
						episode.step_records[step].q_comm_max_t[:, agent_idx] = comm_value

					step += 1

			self.game.step(_action)
			_step += 1
			episode.step_records[step].s_t[:, agent_idx] = torch.tensor(self.game.get_state())
		return episode

	def average_reward(self, episode, normalized=True):
		reward = episode.r.sum()
		return float(reward)

	def train(self, agents, reset=True, verbose=False, test_callback=None):
		opt = self.opt
		if reset:
			for agent in agents[1:]:
				agent.reset()

		rewards = []
		for e in range(opt.nepisodes):
			# run episode
			episode = self.run_episode(agents, train_mode=True)
			norm_r = self.average_reward(episode)
			if verbose:
				print('train epoch:', e, 'avg steps:', episode.steps.float().mean().item(), 'avg reward:', norm_r)
			if opt.model_know_share:
				agents[0].learn_from_episode(episode)
			else:
				for agent in agents[0:]:
					agent.learn_from_episode(episode)

			if e % opt.step_test == 0:
				episode = self.run_episode(agents, train_mode=False)
				norm_r = self.average_reward(episode)
				rewards.append(norm_r)
				if test_callback:
					test_callback(e, norm_r)
				print('TEST EPOCH:', e, 'avg steps:', episode.steps.float().mean().item(), 'avg reward:', norm_r)

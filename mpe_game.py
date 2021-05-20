"""
Switch game

This class manages the state of the Switch game among multiple agents.

RIAL Actions:

1 =
2 =
3 =
4 =
5 =
"""

import numpy as np
import torch

from dotdic import DotDic


class MpeGame:

	def __init__(self, env, opt):
		self.env = env
		self.opt = opt
		self.step_count = 0

		# Set game defaults
		opt_game_default = DotDic({
			'game_action_space': 5,
			'game_reward_shift': 0,
			'game_comm_bits': 1,
			'game_comm_sigma': 2
		})

		for k in opt_game_default:
			if k not in self.opt:
				self.opt[k] = opt_game_default[k]

		self.reset()
		self.observation, self.reward, self.done, self.info = self.env.last()

	def reset(self):
		self.env.reset()

	def get_action_range(self):
		"""
		Return 1-indexed indices into Q vector for valid actions and communications (so 0 represents no-op)
		"""
		opt = self.opt
		action_dtype = torch.long
		action_range = torch.zeros((self.opt.bs, 2), dtype=action_dtype)
		comm_range = torch.zeros((self.opt.bs, 2), dtype=action_dtype)
		for b in range(self.opt.bs):
			action_range[b] = torch.tensor([1, opt.game_action_space], dtype=action_dtype)
			comm_range[b] = torch.tensor([opt.game_action_space + 1, opt.game_action_space_total], dtype=action_dtype)

		return action_range, comm_range

	def step(self, a_t):
		self.env.step(a_t)
		self.step_count += 1

	def get_reward(self):
		self.observation, self.reward, self.done, self.info = self.env.last()

		return self.reward, self.done

	def get_state(self):
		return self.observation

# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple example of setting up a multi-agent version of GFootball with rllib.
"""
import gfootball.env as football_env
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class RllibGFootball(MultiAgentEnv):
	"""An example of a wrapper for GFootball to make it compatible with rllib."""

	def __init__(self, config):
		num_agents = config.get('num_agents', 3) # Simple environment with `num_agents` independent players
		self.env = football_env.create_environment(**config)
		self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
		self.observation_space = gym.spaces.Box(
			low=self.env.observation_space.low[0],
			high=self.env.observation_space.high[0],
			dtype=self.env.observation_space.dtype)
		self.num_agents = num_agents

	def reset(self):
		# original_obs = self.env.reset()
		# obs = {}
		# for x in range(self.num_agents):
		# 	if self.num_agents > 1:
		# 		obs['agent_%d' % x] = original_obs[x]
		# 	else:
		# 		obs['agent_%d' % x] = original_obs
		# return obs
		return self.env.reset()

	def step(self, action_dict):
		actions = [
			value
			for key, value in sorted(action_dict.items())
		]
		# o, r, d, i = self.env.step(actions)
		# rewards = {}
		# obs = {}
		# infos = {}
		# for pos, key in enumerate(sorted(action_dict.keys())):
		# 	infos[key] = i
		# 	if self.num_agents > 1:
		# 		rewards[key] = r[pos]
		# 		obs[key] = o[pos]
		# 	else:
		# 		rewards[key] = r
		# 		obs[key] = o
		# dones = {'__all__': d}
		obs, rewards, has_done, infos = self.env.step(actions)
		dones = {'__all__': has_done}
		return obs, rewards, dones, infos

import logging
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent

import gym
import numpy as np

from environments.primal.Env_Builder import *
from environments.primal.od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError
from environments.primal.od_mstar3 import od_mstar
from environments.primal.GroupLock import Lock
import random

from environments.primal.Primal2Observer import Primal2Observer
from environments.primal.Primal2Env import Primal2Env
from environments.primal.Map_Generator import *


class Primal(MultiAgentEnv):
	metadata = Primal2Env.metadata

	@staticmethod
	def preprocess_observation_dict(obs_dict):
		return {
			k: {
				'map': state,
				'goal': vector,
			}
			for k,(state,vector) in obs_dict.items()
		}

	@property
	def observation_space(self) -> gym.spaces.Space:
		return gym.spaces.Dict({
			'map': gym.spaces.Box(low=-255, high=255, shape=(11, self.observation_size, self.observation_size), dtype=np.float32),
			'goal': gym.spaces.Box(low=-255, high=255, shape=(3,), dtype=np.float32),
		})

	@property
	def action_space(self):
		return gym.spaces.Discrete(9 if self.IsDiagonal else 5)


	def __init__(self, config):
		super().__init__()
		self.observation_size = config.get('observation_size',3)
		self.observer = Primal2Observer(
			observation_size=self.observation_size, 
			num_future_steps=config.get('num_future_steps',3),
		)
		self.map_generator = maze_generator(
			env_size = config.get('env_size',(10, 30)), 
			wall_components = config.get('wall_components',(3, 8)),
			obstacle_density = config.get('obstacle_density',(0.5, 0.7)),
		)
		self.num_agents = config.get('num_agents',None)
		self.IsDiagonal = config.get('IsDiagonal',False)
		self.frozen_steps = config.get('frozen_steps',0)
		self.isOneShot = config.get('isOneShot',False)
		
		self._env = Primal2Env(observer=self.observer, map_generator=self.map_generator, num_agents=self.num_agents, IsDiagonal=self.IsDiagonal, frozen_steps=self.frozen_steps, isOneShot=self.isOneShot)
		self._agent_ids = list(range(1, self.num_agents + 1))
	
	def reset(self):
		self._env._reset()
		obs = self._env._observe()
		# print(obs[1][0].shape, obs[1][1].shape)
		return self.preprocess_observation_dict(obs)

	def get_why_explanation(self, new_pos, old_mstar_pos, old_astar_pos=None):
		explanation_list = []
		# print(new_pos, old_astar_pos)
		if old_astar_pos and new_pos == old_astar_pos:
			explanation_list.append('acting_as_A*')
		if new_pos == old_mstar_pos:
			explanation_list.append('acting_as_M*')
		if not explanation_list:
			explanation_list = ['acting_differently']
		return explanation_list

	# Executes an action by an agent
	def step(self, action_dict):
		# print(action_dict[1])
		astar_pos_dict = {
			i: self._env.expert_until_first_goal(agent_ids=[i])[0][0]
			# i: None
			for i in self._agent_ids
		}
		path_list = self._env.expert_until_first_goal(agent_ids=self._agent_ids)
		mstar_pos_dict = {
			i: path_list[e][0] if path_list and len(path_list) == len(self._agent_ids) else None
			for e,i in enumerate(self._agent_ids)
		}

		obs,rew = self._env.step_all(action_dict)
		obs = self.preprocess_observation_dict(obs)

		# print('qwqw', step_r.obs[0].shape)
		done = {
			k: False #self._env.world.getDone(k) 
			for k in obs.keys()
		}

		positions = self._env.getPositions()
		rew = {
			k: 1 if self._env.isStandingOnGoal[k] else 0
			for k in self._agent_ids
		}
		# throughput = sum((1 if self._env.isStandingOnGoal[k] else 0 for k in self._agent_ids))
		info = {
			k: {
				'explanation': {
					'why': self.get_why_explanation(positions[k], mstar_pos_dict[k], old_astar_pos=astar_pos_dict[k])
				},
				# 'stats_dict': {
				# 	"throughput": throughput
				# }
			}
			for k in self._agent_ids
		}
		
		# print(info)
		done["__all__"] = False #terminal = all(done.values())
		# rew["__all__"] = np.sum([r for r in step_r.reward.values()])
		return obs, rew, done, info

	def render(self,mode='human'):
		return self._env._render(self._env_config.get('render'))

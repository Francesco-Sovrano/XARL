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
				'cnn': state,
				'fc': vector,
			}
			for k,(state,vector) in obs_dict.items()
		}

	@property
	def observation_space(self) -> gym.spaces.Space:
		return gym.spaces.Dict({
			'cnn': gym.spaces.Box(low=-1, high=1, shape=(11, self.observation_size, self.observation_size), dtype=np.float32),
			'fc': gym.spaces.Box(low=-1, high=255, shape=(3,), dtype=np.float32),
		})

	@property
	def action_space(self):
		return gym.spaces.Discrete(9 if self.IsDiagonal else 5)


	def __init__(self, config):
		super().__init__()
		self.observation_size = config.get('observation_size',3)
		self.observer = Primal2Observer(self.observation_size)
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
	
	def reset(self):
		self._env._reset()
		obs = self._env._observe(list(range(1, self.num_agents + 1)))
		# print(obs[1][0].shape, obs[1][1].shape)
		return self.preprocess_observation_dict(obs)

	# Executes an action by an agent
	def step(self, action_dict):
		obs,rew = self._env.step_all(action_dict)

		obs = self.preprocess_observation_dict(obs)
		# print('qwqw', step_r.obs[0].shape)
		done = {
			k: False 
			for k in obs.keys()
		}
		info = {
			k: {
				'explanation': 'change_me_pls'
			} 
			for k in obs.keys()
		}
		# print(info)
		done["__all__"] = False # all(done.values())
		# rew["__all__"] = np.sum([r for r in step_r.reward.values()])
		return obs, rew, done, info

	def render(self,mode='human'):
		return self._env._render(self._env_config.get('render'))

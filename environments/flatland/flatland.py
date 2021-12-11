import logging
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent

import gym
import numpy as np
from flatland.envs.malfunction_generators import no_malfunction_generator, malfunction_from_params, MalfunctionParameters
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from environments.flatland.lib import get_generator_config
from environments.flatland.lib.observations import make_obs

from environments.flatland.lib.utils.gym_env import FlatlandGymEnv, StepOutput
from environments.flatland.lib.utils.gym_env_wrappers import SkipNoChoiceCellsWrapper, AvailableActionsWrapper, \
	ShortestPathActionWrapper, SparseRewardWrapper, DeadlockWrapper, DeadlockResolutionWrapper


class Flatland(MultiAgentEnv):
	reward_range = (-float('inf'), float('inf'))
	spec = None
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 10,
		'semantics.autoreset': True
	}

	@staticmethod
	def preprocess_observation_dict(obs_dict):
		return {
			k: {
				'cnn': obs,
			}
			for k,obs in obs_dict.items()
		}

	def render(self,mode='human'):
		return self._env.render(self._env_config.get('render'))

	def __init__(self, env_config):
		super().__init__()
		self._observation = make_obs(env_config['observation'], env_config.get('observation_config'))
		# print('wawa', self._observation.observation_space())
		self._config = get_generator_config(env_config['generator_config'])

		# Overwrites with env_config seed if it exists
		if env_config.get('seed'):
			self._config['seed'] = env_config.get('seed')

		self._env = FlatlandGymEnv(
			rail_env=self._launch(),
			observation_space=self._observation.observation_space(),
			regenerate_rail_on_reset=self._config['regenerate_rail_on_reset'],
			regenerate_schedule_on_reset=self._config['regenerate_schedule_on_reset']
		)
		if env_config['observation'] == 'shortest_path':
			self._env = ShortestPathActionWrapper(self._env)
		if env_config.get('sparse_reward', False):
			self._env = SparseRewardWrapper(self._env, finished_reward=env_config.get('done_reward', 1),
											not_finished_reward=env_config.get('not_finished_reward', -1))
		if env_config.get('deadlock_reward', 0) != 0:
			self._env = DeadlockWrapper(self._env, deadlock_reward=env_config['deadlock_reward'])
		if env_config.get('resolve_deadlocks', False):
			deadlock_reward = env_config.get('deadlock_reward', 0)
			self._env = DeadlockResolutionWrapper(self._env, deadlock_reward)
		if env_config.get('skip_no_choice_cells', False):
			self._env = SkipNoChoiceCellsWrapper(self._env, env_config.get('accumulate_skipped_rewards', False))
		if env_config.get('available_actions_obs', False):
			self._env = AvailableActionsWrapper(self._env)

	def _launch(self):
		rail_generator = sparse_rail_generator(
			seed=self._config['seed'],
			max_num_cities=self._config['max_num_cities'],
			grid_mode=self._config['grid_mode'],
			max_rails_between_cities=self._config['max_rails_between_cities'],
			max_rails_in_city=self._config['max_rails_in_city']
		)

		malfunction_generator = no_malfunction_generator()
		if {'malfunction_rate', 'min_duration', 'max_duration'} <= self._config.keys():
			stochastic_data = MalfunctionParameters(
				malfunction_rate=self._config['malfunction_rate'],
				min_duration=self._config['malfunction_min_duration'],
				max_duration=self._config['malfunction_max_duration']
			)
			malfunction_generator = malfunction_from_params(stochastic_data)

		speed_ratio_map = None
		if 'speed_ratio_map' in self._config:
			speed_ratio_map = {
				float(k): float(v) for k, v in self._config['speed_ratio_map'].items()
			}
		schedule_generator = sparse_schedule_generator(speed_ratio_map)

		env = None
		try:
			env = RailEnv(
				width=self._config['width'],
				height=self._config['height'],
				rail_generator=rail_generator,
				schedule_generator=schedule_generator,
				number_of_agents=self._config['number_of_agents'],
				malfunction_generator_and_process_data=malfunction_generator,
				obs_builder_object=self._observation.builder(),
				remove_agents_at_target=False,
				random_seed=self._config['seed']
			)

			env.reset()
		except ValueError as e:
			logging.error("=" * 50)
			logging.error(f"Error while creating env: {e}")
			logging.error("=" * 50)

		return env

	# Executes an action by an agent
	def step(self, action_dict):
		step_r = self._env.step(action_dict)

		obs = self.preprocess_observation_dict(step_r.obs)
		# print('qwqw', step_r.obs[0].shape)
		rew = step_r.reward
		done = step_r.done
		info = step_r.info
		# print(info)
		done["__all__"] = all(step_r.done.values())
		# rew["__all__"] = np.sum([r for r in step_r.reward.values()])
		return obs, rew, done, info

	def reset(self):
		# print(self._env.reset()[0].shape)
		return self.preprocess_observation_dict(self._env.reset())

	@property
	def observation_space(self) -> gym.spaces.Space:
		return gym.spaces.Dict({
			'cnn': self._env.observation_space
		})

	@property
	def action_space(self):
		return self._env.action_space

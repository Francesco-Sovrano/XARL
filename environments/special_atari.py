from gym.envs.atari.atari_env import AtariEnv
import numpy as np

class SpecialAtariEnv(AtariEnv):

	def __init__(self, explanation_fn='rewards_only_explanation', **args):
		super().__init__(**args)
		print('explanation_fn:', explanation_fn)
		self.explanation_fn = eval('self.'+explanation_fn)

	def reset(self):
		obs = super().reset()
		self.lives = self.ale.lives()
		return obs

	@staticmethod
	def rewards_only_explanation(reward, terminal, new_ram, old_ram, new_lives, old_lives):
		return [new_ram-old_ram if reward != 0 else 'no_reward']

	@staticmethod
	def rewards_n_lives_explanation(reward, terminal, new_ram, old_ram, new_lives, old_lives):
		lost_lives = old_lives-new_lives
		explanation_list = []
		if reward != 0:
			explanation_list.append(new_ram-old_ram)
		else:
			explanation_list.append('no_reward')
		if lost_lives != 0:
			explanation_list.append('lost_lives')
		return explanation_list
	
	def step(self, a):
		old_lives = self.lives
		old_ram = self._get_ram()
		state, reward, terminal, info_dict = super().step(a)
		new_ram = self._get_ram()
		new_lives = self.ale.lives()

		info_dict['explanation'] = self.explanation_fn(reward, terminal, new_ram, old_ram, new_lives, old_lives)

		self.lives = new_lives
		# print(reward, terminal, delta)
		return state, reward, terminal, info_dict


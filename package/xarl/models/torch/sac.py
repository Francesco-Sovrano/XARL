from ray.rllib.agents.sac.sac_torch_model import SACTorchModel
import gym
import numpy as np
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

class TorchAdaptiveMultiHeadNet:

	@staticmethod
	def init(preprocessing_model):
		class TorchAdaptiveMultiHeadNetInner(SACTorchModel):
			"""
			Data flow:
			`obs` -> forward() (should stay a noop method!) -> `model_out`
			`model_out` -> get_policy_output() -> pi(actions|obs)
			`model_out`, `actions` -> get_q_values() -> Q(s, a)
			`model_out`, `actions` -> get_twin_q_values() -> Q_twin(s, a)
			"""

			def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
				self.policy_preprocessing_model = preprocessing_model(obs_space, self.model_config['custom_model_config'])
				self.policy_preprocessed_obs_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(self.policy_preprocessing_model.get_num_outputs(),), dtype=np.float32)
				return super().build_policy_model(self.policy_preprocessed_obs_space, num_outputs, policy_model_config, name)

			def build_q_model(self, obs_space, action_space, num_outputs, q_model_config, name):
				self.value_preprocessing_model = preprocessing_model(obs_space, self.model_config['custom_model_config'])
				self.value_preprocessed_obs_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(self.value_preprocessing_model.get_num_outputs(),), dtype=np.float32)
				return super().build_q_model(self.value_preprocessed_obs_space, action_space, num_outputs, q_model_config, name)

			def get_policy_output(self, model_out):
				model_out = self.policy_preprocessing_model(model_out)
				return super().get_policy_output(model_out)

			def _get_q_value(self, model_out, actions, net):
				model_out = self.value_preprocessing_model(model_out)
				return super()._get_q_value(model_out, actions, net)

			def policy_variables(self):
				# print(len(self.policy_preprocessing_model.variables()), len(super().policy_variables()))
				return self.policy_preprocessing_model.variables() + super().policy_variables()

			def q_variables(self):
				return self.value_preprocessing_model.variables() + super().q_variables()

		return TorchAdaptiveMultiHeadNetInner

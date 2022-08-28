from ray.rllib.agents.sac.sac_torch_model import SACTorchModel
import gym
import numpy as np
from ray.rllib.utils.framework import try_import_torch
import os

torch, nn = try_import_torch()
# torch.set_num_threads(os.cpu_count())
# torch.set_num_interop_threads(os.cpu_count())

class TorchAdaptiveMultiHeadNet:

	@staticmethod
	def init(policy_preprocessing_model,value_preprocessing_model):
		class TorchAdaptiveMultiHeadNetInner(SACTorchModel):
			"""
			Data flow:
			`obs` -> forward() (should stay a noop method!) -> `model_out`
			`model_out` -> get_policy_output() -> pi(actions|obs)
			`model_out`, `actions` -> get_q_values() -> Q(s, a)
			`model_out`, `actions` -> get_twin_q_values() -> Q_twin(s, a)
			"""

			def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
				self.preprocessing_model_policy = policy_preprocessing_model(obs_space, self.model_config['custom_model_config'])
				preprocessed_obs_space_policy = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(self.preprocessing_model_policy.get_num_outputs(),), dtype=np.float32)
				return super().build_policy_model(preprocessed_obs_space_policy, num_outputs, policy_model_config, name)

			def build_q_model(self, obs_space, action_space, num_outputs, q_model_config, name):
				self.preprocessing_model_q = value_preprocessing_model(obs_space, self.model_config['custom_model_config'])
				preprocessed_obs_space_q = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(self.preprocessing_model_q.get_num_outputs(),), dtype=np.float32)
				return super().build_q_model(preprocessed_obs_space_q, action_space, num_outputs, q_model_config, name)

			def get_policy_output(self, model_out):
				model_out = self.preprocessing_model_policy(model_out)
				return super().get_policy_output(model_out)

			def get_q_values(self, model_out, actions = None):
				model_out = self.preprocessing_model_q(model_out)
				return self._get_q_value(model_out, actions, self.q_net)

			def get_twin_q_values(self, model_out, actions = None):
				model_out = self.preprocessing_model_q(model_out)
				return self._get_q_value(model_out, actions, self.twin_q_net)

			def policy_variables(self):
				return self.preprocessing_model_policy.variables() + super().policy_variables()

			def q_variables(self):
				return self.preprocessing_model_q.variables() + super().q_variables()

		return TorchAdaptiveMultiHeadNetInner

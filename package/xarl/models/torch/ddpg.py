import gym
import os
import numpy as np

from ray.rllib.agents.ddpg.ddpg_torch_model import DDPGTorchModel
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

torch, nn = try_import_torch()
# torch.set_num_threads(os.cpu_count())
# torch.set_num_interop_threads(os.cpu_count())

class TorchAdaptiveMultiHeadDDPG:

	@staticmethod
	def init(preprocessing_model):
		class TorchAdaptiveMultiHeadDDPGInner(DDPGTorchModel):

			def __init__(self, obs_space, action_space, num_outputs, model_config, name, actor_hiddens=None, actor_hidden_activation="relu", critic_hiddens=None, critic_hidden_activation="relu", twin_q=False, add_layer_norm=False):
				num_outputs = preprocessing_model(obs_space, model_config['custom_model_config']).get_num_outputs()
				super().__init__(obs_space, action_space, num_outputs, model_config, name, actor_hiddens, actor_hidden_activation, critic_hiddens, critic_hidden_activation, twin_q, add_layer_norm)
				self.preprocessing_model_policy = preprocessing_model(obs_space, model_config['custom_model_config'])
				self.preprocessing_model_q = preprocessing_model(obs_space, model_config['custom_model_config'])

			def forward(self, input_dict, state, seq_lens):
				return input_dict["obs"], state

			def get_policy_output(self, model_out):
				model_out = self.preprocessing_model_policy(model_out)
				return self.policy_model(model_out)

			def get_q_values(self, model_out, actions = None):
				model_out = self.preprocessing_model_q(model_out)
				return self.q_model(torch.cat([model_out, actions], -1))

			def get_twin_q_values(self, model_out, actions = None):
				model_out = self.preprocessing_model_q(model_out)
				return self.twin_q_model(torch.cat([model_out, actions], -1))

			def policy_variables(self, as_dict=False):
				if not as_dict:
					return self.preprocessing_model_policy.variables(as_dict) + super().policy_variables(as_dict)
				p_dict = super().policy_variables(as_dict)
				p_dict.update(self.preprocessing_model_policy.variables(as_dict))
				return p_dict

			def q_variables(self, as_dict=False):
				if not as_dict:
					return self.preprocessing_model_q.variables(as_dict) + super().q_variables(as_dict)
				q_dict = super().q_variables(as_dict)
				q_dict.update(self.preprocessing_model_q.variables(as_dict))
				return q_dict

		return TorchAdaptiveMultiHeadDDPGInner

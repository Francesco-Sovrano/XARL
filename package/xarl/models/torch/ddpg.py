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
				if actor_hiddens is None:
					actor_hiddens = [256, 256]

				if critic_hiddens is None:
					critic_hiddens = [256, 256]

				# old_obs_space = obs_space
				# old_num_outputs = num_outputs
				num_outputs = preprocessing_model(obs_space, model_config['custom_model_config']).get_num_outputs()
				# obs_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(num_outputs,), dtype=np.float32)
				
				nn.Module.__init__(self)
				super(DDPGTorchModel, self).__init__(
					obs_space, action_space, num_outputs, model_config, name
				)
				
				self.bounded = np.logical_and(
					self.action_space.bounded_above, self.action_space.bounded_below
				).any()
				self.action_dim = np.product(self.action_space.shape)

				# Build the policy network.
				self.preprocessing_model_policy_model = preprocessing_model(obs_space, model_config['custom_model_config'])
				self.policy_model = nn.Sequential()
				ins = num_outputs
				self.obs_ins = ins
				activation = get_activation_fn(actor_hidden_activation, framework="torch")
				for i, n in enumerate(actor_hiddens):
					self.policy_model.add_module(
						"action_{}".format(i),
						SlimFC(
							ins,
							n,
							initializer=torch.nn.init.xavier_uniform_,
							activation_fn=activation,
						),
					)
					# Add LayerNorm after each Dense.
					if add_layer_norm:
						self.policy_model.add_module(
							"LayerNorm_A_{}".format(i), nn.LayerNorm(n)
						)
					ins = n

				self.policy_model.add_module(
					"action_out",
					SlimFC(
						ins,
						self.action_dim,
						initializer=torch.nn.init.xavier_uniform_,
						activation_fn=None,
					),
				)

				# Use sigmoid to scale to [0,1], but also double magnitude of input to
				# emulate behaviour of tanh activation used in DDPG and TD3 papers.
				# After sigmoid squashing, re-scale to env action space bounds.
				class _Lambda(nn.Module):
					def __init__(self_):
						super().__init__()
						low_action = nn.Parameter(
							torch.from_numpy(self.action_space.low).float()
						)
						low_action.requires_grad = False
						self_.register_parameter("low_action", low_action)
						action_range = nn.Parameter(
							torch.from_numpy(
								self.action_space.high - self.action_space.low
							).float()
						)
						action_range.requires_grad = False
						self_.register_parameter("action_range", action_range)

					def forward(self_, x):
						sigmoid_out = nn.Sigmoid()(2.0 * x)
						squashed = self_.action_range * sigmoid_out + self_.low_action
						return squashed

				# Only squash if we have bounded actions.
				if self.bounded:
					self.policy_model.add_module("action_out_squashed", _Lambda())

				# Build the Q-net(s), including target Q-net(s).
				def build_q_net(name_):
					activation = get_activation_fn(critic_hidden_activation, framework="torch")
					# For continuous actions: Feed obs and actions (concatenated)
					# through the NN. For discrete actions, only obs.
					q_net = nn.Sequential()
					ins = self.obs_ins + self.action_dim
					for i, n in enumerate(critic_hiddens):
						q_net.add_module(
							"{}_hidden_{}".format(name_, i),
							SlimFC(
								ins,
								n,
								initializer=torch.nn.init.xavier_uniform_,
								activation_fn=activation,
							),
						)
						ins = n

					q_net.add_module(
						"{}_out".format(name_),
						SlimFC(
							ins,
							1,
							initializer=torch.nn.init.xavier_uniform_,
							activation_fn=None,
						),
					)
					return q_net

				self.preprocessing_model_q = preprocessing_model(obs_space, model_config['custom_model_config'])
				self.q_model = build_q_net("q")
				if twin_q:
					self.preprocessing_model_twin_q = preprocessing_model(obs_space, model_config['custom_model_config'])
					self.twin_q_model = build_q_net("twin_q")
				else:
					self.twin_q_model = None

			def forward(self, input_dict, state, seq_lens):
				return input_dict["obs"], state

			def get_policy_output(self, model_out):
				model_out = self.preprocessing_model_policy_model(model_out)
				return self.policy_model(model_out)

			def get_q_values(self, model_out, actions = None):
				model_out = self.preprocessing_model_q(model_out)
				return self.q_model(torch.cat([model_out, actions], -1))

			def get_twin_q_values(self, model_out, actions = None):
				model_out = self.preprocessing_model_twin_q(model_out)
				return self.twin_q_model(torch.cat([model_out, actions], -1))

			def policy_variables(self, as_dict=False):
				if not as_dict:
					return self.preprocessing_model_policy_model.variables(as_dict) + super().policy_variables(as_dict)
				p_dict = super().policy_variables(as_dict)
				p_dict.update(self.preprocessing_model_policy_model.variables(as_dict))
				return p_dict

			def q_variables(self, as_dict=False):
				if not as_dict:
					return self.preprocessing_model_q.variables(as_dict) + (self.preprocessing_model_twin_q.variables(as_dict) if self.twin_q_model else []) + super().q_variables(as_dict)
				q_dict = super().q_variables(as_dict)
				q_dict.update(self.preprocessing_model_q.variables(as_dict))
				if self.twin_q_model:
					q_dict.update(self.preprocessing_model_twin_q.variables(as_dict))
				return q_dict

		return TorchAdaptiveMultiHeadDDPGInner

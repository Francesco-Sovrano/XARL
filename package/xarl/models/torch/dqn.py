from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

class TorchAdaptiveMultiHeadDQN:

	@staticmethod
	def init(preprocessing_model):
		class TorchAdaptiveMultiHeadDQNInner(DQNTorchModel):
			def __init__(self, obs_space, action_space, num_outputs, model_config, name, *, q_hiddens = (256,), dueling = False, dueling_activation = "relu", num_atoms = 1, use_noisy = False, v_min = -10.0, v_max = 10.0, sigma0 = 0.5, add_layer_norm = False):
				tmp_model = preprocessing_model(obs_space, model_config['custom_model_config']) # do not initialize self.preprocessing_model here or else it won't train
				super().__init__(
					obs_space=obs_space, 
					action_space=action_space, 
					num_outputs=tmp_model.get_num_outputs(), 
					model_config=model_config, 
					name=name, 
					q_hiddens=q_hiddens, 
					dueling=dueling, 
					dueling_activation=dueling_activation,
					num_atoms=num_atoms, 
					use_noisy=use_noisy, 
					v_min=v_min, 
					v_max=v_max, 
					sigma0=sigma0, 
					add_layer_norm=add_layer_norm
				)
				self.preprocessing_model = preprocessing_model(obs_space, model_config['custom_model_config'])

			def forward(self, input_dict, state, seq_lens):
				model_out = self.preprocessing_model(input_dict['obs'])
				return model_out, state

		return TorchAdaptiveMultiHeadDQNInner

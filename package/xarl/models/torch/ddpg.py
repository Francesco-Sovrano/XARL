from ray.rllib.agents.ddpg.ddpg_torch_model import DDPGTorchModel
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

class TorchAdaptiveMultiHeadDDPG:

	@staticmethod
	def init(preprocessing_model):
		class TorchAdaptiveMultiHeadDDPGInner(DDPGTorchModel):
			def __init__(self, obs_space, action_space, num_outputs, model_config, name, actor_hiddens=(256, 256), actor_hidden_activation="relu", critic_hiddens=(256, 256), critic_hidden_activation="relu", twin_q=False, add_layer_norm=False):
				nn.Module.__init__(self)
				self.preprocessing_model = preprocessing_model(obs_space, model_config)
				# self.register_variables(self.preprocessing_model.variables)
				super().__init__(
					obs_space=obs_space, 
					action_space=action_space, 
					num_outputs=self.preprocessing_model.get_num_outputs(), 
					model_config=model_config, 
					name=name, 
					actor_hiddens=actor_hiddens, 
					actor_hidden_activation=actor_hidden_activation, 
					critic_hiddens=critic_hiddens, 
					critic_hidden_activation=critic_hidden_activation, 
					twin_q=twin_q, 
					add_layer_norm=add_layer_norm
				)

			def forward(self, input_dict, state, seq_lens):
				model_out = self.preprocessing_model(input_dict['obs'])
				return model_out, state

		return TorchAdaptiveMultiHeadDDPGInner

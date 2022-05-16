from xarl.models.torch.head_generator.adaptive_model_wrapper import AdaptiveModel
from xarl.utils.torch_gnn_models.model import GNNBranch
from ray.rllib.utils.framework import try_import_torch
import logging
import numpy as np
import gym

logger = logging.getLogger(__name__)
torch, nn = try_import_torch()

class CommAdaptiveModel(AdaptiveModel):
	def __init__(self, obs_space, config):
		# print('CommAdaptiveModel', config)
		if hasattr(obs_space, 'original_space'):
			obs_space = obs_space.original_space
		logger.warning(f"Building keras layers for Comm model with {len(obs_space['all_agents_features_list'])} agents")
		super().__init__(obs_space['all_agents_features_list'][0], config)
		self.obs_space = obs_space

		#################
		###### GNN ######
		agent_features_size = self.get_agent_features_size()
		logger.warning(f"Agent features size: {agent_features_size}")
		message_features = config.get('message_features', 32)
		self.comm_range = torch.Tensor([config.get('comm_range', 10.)])
		self.gnn = GNNBranch(
			in_features= agent_features_size, 
			msg_features= message_features, 
			out_features= message_features, 
			activation= config.get('gnn_activation', 'relu')
		)
		# self.use_beta = True

	def get_agent_features_size(self):
		def get_random_input_recursively(_obs_space):
			if isinstance(_obs_space, gym.spaces.Dict):
				return {
					k: get_random_input_recursively(v)
					for k,v in _obs_space.spaces.items()
				}
			elif isinstance(_obs_space, gym.spaces.Tuple):
				return list(map(get_random_input_recursively, _obs_space.spaces))
			return torch.rand(1,*_obs_space.shape)
		random_obs = get_random_input_recursively(self.obs_space['all_agents_features_list'][0])
		return super().forward(random_obs).data.shape[-1]

	def forward(self, x):
		super_forward = super().forward
		
		this_agent_id_mask = x['this_agent_id_mask'][:,:,None] # add extra dimension
		all_agents_positions = torch.stack(x['all_agents_position_list'], dim=1)
		all_agents_features = torch.stack(list(map(super_forward, x['all_agents_features_list'])), dim=1)

		# main_output = torch.sum(all_agents_features*this_agent_id_mask, dim=1)
		
		gnn_output = self.gnn(all_agents_positions, all_agents_features, self.comm_range)
		message_from_others = torch.sum(gnn_output*this_agent_id_mask, dim=1)
		
		output = message_from_others
		# output = torch.cat([main_output, message_from_others], dim=1)
		return output

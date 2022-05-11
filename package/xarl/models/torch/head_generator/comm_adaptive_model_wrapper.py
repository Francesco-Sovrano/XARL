from xarl.models.torch.head_generator.adaptive_model_wrapper import AdaptiveModel, get_input_shape_recursively
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
		logger.warning(f"Building keras layers for Comm model with {len(obs_space['all_agents'])} agents")
		super().__init__(obs_space['this_agent'], config)
		self.obs_space = obs_space

		#################
		###### GNN ######
		# print(get_input_shape_recursively(obs_space['all_agents'][0]['features']))
		# agent_features_size = sum((np.prod(shape) for shape in get_input_shape_recursively(obs_space['all_agents'][0]['features'])))
		agent_features_size = self.get_agent_features_size()
		logger.warning(f"Agent features size: {agent_features_size}")
		message_features = config.get('message_features', 32)
		self.comm_range = torch.Tensor([config.get('comm_range', 2.)])
		self.gnn = GNNBranch(
			in_features= agent_features_size, 
			msg_features= message_features, 
			out_features= message_features, 
			activation= config.get('activation', 'relu')
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
		random_obs = get_random_input_recursively(self.obs_space['all_agents'][0]['features'])
		return super().forward(random_obs).data.shape[-1]

	def build_message_old(self, x):
		others_output_list = [super().forward(y['features']) for y in x['all_agents']]
		###	Apply visibility mask
		message_visibility_mask_input = x['message_visibility_mask']
		masked_messages = self.apply_visibility_mask(
			others_output_list, 
			message_visibility_mask_input
		)
		####
		return sum(masked_messages)

	@staticmethod
	def apply_visibility_mask(feature_list, message_visibility_mask_input):
		return [
			m*message_visibility_mask_input[:,i][:,None]
			for i,m in enumerate(feature_list)
		]

	def forward(self, x):
		super_forward = super().forward
		main_output = super_forward(x['this_agent'])

		message_visibility_mask_input = x['message_visibility_mask']
		agent_position = torch.stack(
			self.apply_visibility_mask(
				(y['position'] for y in x['all_agents']), 
				message_visibility_mask_input
			), 
			dim=1
		)
		# print(agent_position.shape)
		# agent_position = torch.flatten(agent_position, start_dim=1)
		agent_features = torch.stack(
			self.apply_visibility_mask(
				(super_forward(y['features']) for y in x['all_agents']), 
				message_visibility_mask_input
			), 
			dim=1
		)
		# print(agent_features.shape)
		# agent_features = torch.flatten(agent_features, start_dim=1)
		others_output = self.gnn(agent_position, agent_features, self.comm_range)
		message_from_others = torch.flatten(others_output, start_dim=1)
		
		output = torch.cat([main_output, message_from_others], dim=1)
		return output

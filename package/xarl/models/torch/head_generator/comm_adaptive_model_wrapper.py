from xarl.models.torch.head_generator.adaptive_model_wrapper import AdaptiveModel
from xarl.utils.torch_gnn_models.model import GNNBranch
from ray.rllib.utils.framework import try_import_torch
import logging
import numpy as np
import gym
import torch_geometric

logger = logging.getLogger(__name__)
torch, nn = try_import_torch()

class CommAdaptiveModel(AdaptiveModel):
	def __init__(self, obs_space, config):
		# print('CommAdaptiveModel', config)
		if hasattr(obs_space, 'original_space'):
			obs_space = obs_space.original_space
		super().__init__(obs_space['all_agents_relative_features_list'][0], config)
		self.obs_space = obs_space

		#################
		###### GNN ######
		agent_features_size = self.get_agent_features_size()
		logger.warning(f"Agent features size: {agent_features_size}")
		self.n_agents = len(obs_space['all_agents_absolute_position_list'])
		self.n_leaders = len(obs_space['all_leaders_absolute_position_list']) if 'all_leaders_absolute_position_list' in obs_space else 0
		self.n_agents_and_leaders = self.n_agents + self.n_leaders
		self.max_num_neighbors = config.get('max_num_neighbors', 32)
		self.message_size = config.get('message_size', 32)
		self.comm_range = torch.Tensor([config.get('comm_range', 10.)])
		self.gnn = GNNBranch(
			node_features=agent_features_size,
			edge_features=2, # position
			out_features=self.message_size,
			node_embedding=config.get('node_embedding_units', 8),
			edge_embedding=config.get('edge_embedding_units', 8),
			gnn_embedding=config.get('gnn_embedding_units', 32),
		)
		logger.warning(f"Building keras layers for Comm model with {self.n_agents} agents, {self.n_leaders} leaders and communication range {self.comm_range[0]} for maximum {self.max_num_neighbors} neighbours")
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
		random_obs = get_random_input_recursively(self.obs_space['all_agents_relative_features_list'][0])
		return super().forward(random_obs).data.shape[-1]

	def forward(self, x):
		super_forward = super().forward
		
		this_agent_id_mask = x['this_agent_id_mask'][:,:,None] # add extra dimension
		all_agents_features = torch.stack(list(map(super_forward, x['all_agents_relative_features_list'])), dim=1)
		main_output = torch.sum(all_agents_features*this_agent_id_mask, dim=1)

		all_agents_positions = torch.stack(x['all_agents_absolute_position_list'], dim=1)
		if self.n_leaders:
			all_agents_positions = torch.cat(
				[
					torch.stack(x['all_leaders_absolute_position_list'], dim=1), 
					all_agents_positions
				], 
				dim=1
			)

		device = all_agents_positions.device
		batch_size = all_agents_positions.shape[0]

		## build graphs
		graphs = torch_geometric.data.Batch()
		graphs.batch = torch.repeat_interleave(
			torch.arange(batch_size), 
			self.n_agents_and_leaders, 
			dim=0
		).to(device)
		graphs.pos = all_agents_positions.reshape(-1, all_agents_positions.shape[-1])
		graphs.x = all_agents_features.reshape(-1, all_agents_features.shape[-1])
		if self.n_leaders:
			all_agents_types = torch.zeros(batch_size, self.n_agents_and_leaders, 1, device=device)
			all_agents_types[:, :self.n_leaders] = 1.0
			all_agents_types = all_agents_types.reshape(-1, all_agents_types.shape[-1])
			graphs.x = torch.cat([graphs.x, all_agents_types], dim=1)
		graphs = torch_geometric.transforms.RadiusGraph(r=self.comm_range, loop=False, max_num_neighbors=self.max_num_neighbors)(graphs.to(device)) # Creates edges based on node positions pos to all points within a given distance (functional name: radius_graph).
		graphs = torch_geometric.transforms.Cartesian(norm=False)(graphs.to(device)) # Saves the relative Cartesian coordinates of linked nodes in its edge attributes (functional name: cartesian).

		## process graphs
		gnn_output = self.gnn(graphs.x, graphs.edge_index, graphs.edge_attr)
		assert not gnn_output.isnan().any()
		gnn_output = gnn_output.view(-1, self.n_agents_and_leaders, self.message_size) # reshape GNN outputs
		if self.n_leaders:
			gnn_output = gnn_output[:, self.n_leaders:]
		message_from_others = torch.sum(gnn_output*this_agent_id_mask, dim=1)

		## build output
		# output = message_from_others
		output = torch.cat([main_output, message_from_others], dim=1)
		return output

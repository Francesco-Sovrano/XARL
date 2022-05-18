# Code by Jan Blumenkamp

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from torch_cluster import radius_graph

from torch_geometric.transforms import BaseTransform


class GNNConv(MessagePassing):
	propagate_type = {"x": Tensor}

	def __init__(self, nn, aggr="mean", **kwargs):
		super(GNNConv, self).__init__(aggr=aggr, **kwargs)
		self.nn = nn

		self.reset_parameters()

	def reset_parameters(self):
		torch_geometric.nn.inits.reset(self.nn)

	def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
		return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

	def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
		features = torch.cat([edge_attr, x_j], dim=1)
		return self.nn(features)

	def __repr__(self):
		return "{}(nn={})".format(self.__class__.__name__, self.nn)


class RelVel(BaseTransform):
	def __init__(self):
		pass

	def __call__(self, data):
		(row, col), vel, pseudo = data.edge_index, data.vel, data.edge_attr

		cart = vel[row] - vel[col]
		cart = cart.view(-1, 1) if cart.dim() == 1 else cart
		pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
		data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)

		return data

	def __repr__(self) -> str:
		return f"{self.__class__.__name__}(norm={self.norm}, " f"max_value={self.max})"

class GNNBranch(nn.Module):
	def __init__(
		self,
		node_features,
		edge_features,
		out_features,
		node_embedding=32,
		edge_embedding=32,
		gnn_embedding=64,
	):
		nn.Module.__init__(self)

		self.gnn_nn = torch.nn.Sequential(
			torch.nn.Linear(node_embedding + edge_embedding, 16),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(16, gnn_embedding),
			torch.nn.LeakyReLU(),
		)

		'''
		self.gnn = torch_geometric.nn.conv.GATv2Conv(
			in_channels=node_embedding,
			out_channels=gnn_embedding,
			edge_dim=edge_embedding,
			add_self_loops=False,
		).jittable()
		'''

		self.post_gnn = torch.nn.Sequential(
			torch.nn.Linear(gnn_embedding, 16),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(16, gnn_embedding),
			torch.nn.LeakyReLU(),
			#torch.nn.Linear(32, 8),
			#torch.nn.LeakyReLU(),
		)

		'''
		self.gnn = torch_geometric.nn.conv.GINEConv(
			nn=self.post_gnn,
			edge_dim=None,
		).jittable()
		'''

		self.gnn = GNNConv(nn=self.gnn_nn, aggr="add")

		self.node_enc = torch.nn.Sequential(
			torch.nn.Linear(node_features, 16),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(16, node_embedding),
			torch.nn.LeakyReLU(),
			#torch.nn.Linear(32, 32),
			#torch.nn.LeakyReLU(),
			#torch.nn.Linear(32, node_embedding),
		)
		self.edge_enc = torch.nn.Sequential(
			torch.nn.Linear(edge_features, 16),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(16, edge_embedding),
			torch.nn.LeakyReLU(),
			#torch.nn.Linear(32, edge_embedding),
		)
		self.post_proc = torch.nn.Sequential(
			torch.nn.LayerNorm(node_features + gnn_embedding),
			torch.nn.Linear(node_features + gnn_embedding, 32),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(32, 32),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(32, out_features),
		)

	def assert_not_nan(self):
		def assert_not_nan_linear(l):
			assert not l.weight.isnan().any()
			assert not l.bias.isnan().any()

		def assert_not_nan_seq(s):
			for l in s:
				if isinstance(l, torch.nn.Linear):
					assert_not_nan_linear(l)

		assert_not_nan_seq(self.node_enc)
		assert_not_nan_seq(self.edge_enc)
		assert_not_nan_seq(self.post_proc)

	def forward(self, x, edge_index, edge_attr):
		# self.assert_not_nan()
		# node_pre = self.node_pre(pos, vel)

		node_enc = self.node_enc(x)
		edge_enc = self.edge_enc(edge_attr)
		gnn_out = self.gnn(node_enc, edge_index, edge_enc)
		#post_gnn = self.post_gnn(gnn_out)
		#local = self.local(x)
		postproc = torch.cat([gnn_out, x], dim=1)
		return self.post_proc(postproc)


class Model(TorchModelV2, nn.Module):
	def __init__(self, obs_space, action_space, num_outputs, model_config, name, **cfg):
		TorchModelV2.__init__(
			self, obs_space, action_space, num_outputs, model_config, name
		)
		nn.Module.__init__(self)

		self.n_agents = obs_space.original_space["agents_pos"].shape[0]
		self.n_leaders = obs_space.original_space["leaders_pos"].shape[0]
		self.n_agents_leaders = self.n_agents + self.n_leaders
		self.outputs_per_agent = int(num_outputs / self.n_agents)
		self.comm_range = cfg["comm_range"]

		node_embedding=4
		edge_embedding=4
		gnn_embedding=16
		self.gnn = GNNBranch(
			node_features=3,
			edge_features=4,
			out_features=self.outputs_per_agent,
			node_embedding=node_embedding,
			edge_embedding=edge_embedding,
			gnn_embedding=gnn_embedding,
		)
		self.gnn_value = GNNBranch(
			node_features=3,
			edge_features=4,
			out_features=1,
			node_embedding=node_embedding,
			edge_embedding=edge_embedding,
			gnn_embedding=gnn_embedding,
		)

		self.use_beta = False

	@override(ModelV2)
	def forward(self, input_dict, state, seq_lens):
		#agents_pos = input_dict["obs"]["agents_pos"]
		#agents_vel = input_dict["obs"]["agents_vel"]
		#leaders_pos = input_dict["obs"]["leaders_pos"]
		#leaders_vel = input_dict["obs"]["leaders_vel"]
		#leaders_des_vel = input_dict["obs"]["leaders_des_vel"]

		agents_pos = torch.cat([input_dict["obs"]["leaders_pos"], input_dict["obs"]["agents_pos"]], dim=1)
		agents_vel = torch.cat([input_dict["obs"]["leaders_vel"], input_dict["obs"]["agents_vel"]], dim=1)

		assert not agents_pos.isnan().any()
		assert not agents_vel.isnan().any()

		batch_size = agents_pos.shape[0]
		#adj = torch.ones(batch_size, self.n_agents_leaders, self.n_agents_leaders)

		agents_types = torch.zeros(batch_size, self.n_agents_leaders, 1, device=agents_pos.device)
		agents_types[:, :self.n_leaders] = 1.0
		agents_types_flat = agents_types.reshape(batch_size * self.n_agents_leaders, agents_types.shape[-1])

		graphs = torch_geometric.data.Batch()
		graphs.batch = torch.repeat_interleave(
			torch.arange(batch_size), self.n_agents_leaders, dim=0
		).to(agents_pos.device)
		graphs.pos = agents_pos.reshape(-1, agents_pos.shape[-1])
		graphs.vel = agents_vel.reshape(-1, agents_vel.shape[-1])
		graphs.x = torch.cat([graphs.vel, agents_types_flat], dim=1)
		#graphs.edge_index = torch_geometric.utils.dense_to_sparse(adj)[0]

		graphs.edge_index = torch_geometric.nn.pool.radius_graph(
			graphs.pos, batch=graphs.batch, r=self.comm_range, loop=False
		)

		#graphs = torch_geometric.transforms.Distance(norm=False)(
		#    graphs.to(agents_pos.device)
		#)
		graphs = torch_geometric.transforms.Cartesian(norm=False)(graphs.to(agents_pos.device))
		graphs = RelVel()(graphs)

		outputs = self.gnn(graphs.x, graphs.edge_index, graphs.edge_attr)
		values = self.gnn_value(graphs.x, graphs.edge_index, graphs.edge_attr)
		assert not outputs.isnan().any()
		assert not values.isnan().any()

		self._cur_value = values.view(-1, self.n_agents_leaders)[:, self.n_leaders:]
		outputs_agents = outputs.view(-1, self.n_agents_leaders, self.outputs_per_agent)[:, self.n_leaders:]
		return outputs_agents.view(-1, self.n_agents * self.outputs_per_agent), state

	@override(ModelV2)
	def value_function(self):
		assert self._cur_value is not None, "must call forward() first"
		return self._cur_value

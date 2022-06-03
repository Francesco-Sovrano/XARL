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

class GNNBranch(nn.Module):
	def __init__(
		self,
		node_features,
		edge_features,
		out_features,
	):
		nn.Module.__init__(self)
		self.gnn_nn = torch.nn.Sequential(
			torch.nn.Linear(node_features + edge_features, out_features),
			torch.nn.LeakyReLU(),
			# torch.nn.Linear(node_features + edge_features, (node_features + edge_features)//2),
			# torch.nn.LeakyReLU(),
			# torch.nn.Linear((node_features + edge_features)//2, out_features),
			# torch.nn.LeakyReLU(),
		)
		self.gnn = GNNConv(nn=self.gnn_nn, aggr="add")

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
		return self.gnn(x, edge_index, edge_attr)

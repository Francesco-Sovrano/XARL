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
		node_embedding,
		edge_embedding,
		out_features,
	):
		nn.Module.__init__(self)

		self.gnn_nn = torch.nn.Sequential(
			torch.nn.Linear(node_embedding + edge_embedding, out_features),
			torch.nn.LeakyReLU(),
		)

		self.gnn = GNNConv(nn=self.gnn_nn, aggr="add")

		self.node_enc = torch.nn.Sequential(
			torch.nn.Linear(node_features, node_embedding),
			torch.nn.LeakyReLU(),
		)
		self.edge_enc = torch.nn.Sequential(
			torch.nn.Linear(edge_features, edge_embedding),
			torch.nn.LeakyReLU(),
		)

	def forward(self, x, edge_index, edge_attr):
		node_enc = self.node_enc(x)
		edge_enc = self.edge_enc(edge_attr)
		return self.gnn(node_enc, edge_index, edge_enc)

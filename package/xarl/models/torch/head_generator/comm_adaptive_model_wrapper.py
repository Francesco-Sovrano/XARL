from xarl.models.torch.head_generator.adaptive_model_wrapper import AdaptiveModel
from ray.rllib.utils.framework import try_import_torch
import logging

logger = logging.getLogger(__name__)
torch, nn = try_import_torch()

class CommAdaptiveModel(AdaptiveModel):
	def __init__(self, obs_space):
		if hasattr(obs_space, 'original_space'):
			obs_space = obs_space.original_space
		super().__init__(obs_space['this_agent'])
		self.obs_space = obs_space

	def forward(self, x):
		# print(x)
		logger.warning(f"Building keras layers for Comm model with {len(x['all_agents'])} agents")
		main_output = super().forward(x['this_agent'])
		
		###	Apply visibility mask
		apply_obs_to_message_model = super().forward
		others_output_list = list(map(apply_obs_to_message_model,x['all_agents']))
		message_visibility_mask_input = x['message_visibility_mask']
		masked_messages = [
			m*message_visibility_mask_input[:,i]
			for i,m in enumerate(others_output_list)
		]
		###
		others_output = sum(masked_messages)
		output = torch.cat([main_output, others_output], dim=1)
		# print(99, output.shape)
		return output


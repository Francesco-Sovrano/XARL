"""
XADDPG - eXplanation-Aware Deep Deterministic Policy Gradient
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-deterministic-policy-gradients-ddpg-td3
"""  # noqa: E501

from deer.agents.xadqn import XADQNTrainer, XADQN_EXTRA_OPTIONS
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_DEFAULT_CONFIG
from ray.rllib.agents.ddpg.td3 import TD3Trainer, TD3_DEFAULT_CONFIG
from ray.rllib.agents.ddpg.ddpg_tf_policy import DDPGTFPolicy
from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy
from deer.agents.xaddpg.xaddpg_tf_policy import XADDPGTFPolicy
from deer.agents.xaddpg.xaddpg_torch_policy import XADDPGTorchPolicy
from deer.experience_buffers.replay_ops import add_policy_signature

XADDPG_DEFAULT_CONFIG = DDPGTrainer.merge_trainer_configs(
	DDPG_DEFAULT_CONFIG, # For more details, see here: https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
	XADQN_EXTRA_OPTIONS,
	_allow_unknown_configs=True
)

XATD3_DEFAULT_CONFIG = TD3Trainer.merge_trainer_configs(
	TD3_DEFAULT_CONFIG, # For more details, see here: https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
	XADQN_EXTRA_OPTIONS,
	_allow_unknown_configs=True
)

########################
# XADDPG's Execution Plan
########################

class XADDPGTrainer(DDPGTrainer):
	def get_default_config(cls):
		return XADDPG_DEFAULT_CONFIG
		
	def get_default_policy_class(self, config):
		return XADDPGTorchPolicy if config["framework"] == "torch" else XADDPGTFPolicy

	def validate_config(self, config):
		# Call super's validation method.
		super().validate_config(config)

		if config["model"]["custom_model_config"].get("add_nonstationarity_correction", False):
			class PolicySignatureListCollector(SimpleListCollector):
				def get_inference_input_dict(self, policy_id):
					batch = super().get_inference_input_dict(policy_id)
					policy = self.policy_map[policy_id]
					return add_policy_signature(batch,policy)
			config["sample_collector"] = PolicySignatureListCollector
		
	@staticmethod
	def execution_plan(workers, config, **kwargs):
		return XADQNTrainer.execution_plan(workers, config, **kwargs)

class XATD3Trainer(TD3Trainer):
	def get_default_config(cls):
		return XATD3_DEFAULT_CONFIG

	def get_default_policy_class(self, config):
		return XADDPGTorchPolicy if config["framework"] == "torch" else XADDPGTFPolicy

	def validate_config(self, config):
		# Call super's validation method.
		super().validate_config(config)

		if config["model"]["custom_model_config"].get("add_nonstationarity_correction", False):
			class PolicySignatureListCollector(SimpleListCollector):
				def get_inference_input_dict(self, policy_id):
					batch = super().get_inference_input_dict(policy_id)
					policy = self.policy_map[policy_id]
					return add_policy_signature(batch,policy)
			config["sample_collector"] = PolicySignatureListCollector
		
	@staticmethod
	def execution_plan(workers, config, **kwargs):
		return XADQNTrainer.execution_plan(workers, config, **kwargs)
		
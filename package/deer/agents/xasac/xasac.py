"""
XASAC - eXplanation-Aware Soft Actor-Critic
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-deterministic-policy-gradients-ddpg-td3
"""  # noqa: E501

from deer.agents.xadqn import XADQNTrainer, XADQN_EXTRA_OPTIONS
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.agents.sac.sac import SACTrainer, DEFAULT_CONFIG as SAC_DEFAULT_CONFIG
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS
from deer.agents.xasac.xasac_tf_policy import XASACTFPolicy
from deer.agents.xasac.xasac_torch_policy import XASACTorchPolicy
from deer.experience_buffers.replay_ops import add_policy_signature
import copy

XASAC_EXTRA_OPTIONS = copy.deepcopy(XADQN_EXTRA_OPTIONS)
XASAC_EXTRA_OPTIONS["buffer_options"]['clustering_xi'] = 4
XASAC_DEFAULT_CONFIG = SACTrainer.merge_trainer_configs(
	SAC_DEFAULT_CONFIG, # For more details, see here: https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
	XASAC_EXTRA_OPTIONS,
	_allow_unknown_configs=True
)

########################
# XASAC's Policy
########################

class XASACTrainer(SACTrainer):
	def get_default_config(cls):
		return XASAC_DEFAULT_CONFIG

	def get_default_policy_class(self, config):
		return XASACTorchPolicy if config["framework"] == "torch" else XASACTFPolicy

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
		
"""
XASAC - eXplanation-Aware Soft Actor-Critic
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-deterministic-policy-gradients-ddpg-td3
"""  # noqa: E501

from xarl.agents.xadqn import xa_postprocess_nstep_and_prio, xadqn_execution_plan, XADQN_EXTRA_OPTIONS
from ray.rllib.agents.sac.sac import SACTrainer, DEFAULT_CONFIG as SAC_DEFAULT_CONFIG
from ray.rllib.agents.sac.sac_torch_policy import SACTorchPolicy
from ray.rllib.agents.sac.sac_tf_policy import SACTFPolicy
from xarl.agents.xasac.xasac_tf_loss import xasac_actor_critic_loss as tf_xasac_actor_critic_loss
from xarl.agents.xasac.xasac_torch_loss import xasac_actor_critic_loss as torch_xasac_actor_critic_loss
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

XASACTFPolicy = SACTFPolicy.with_updates(
	name="XASACTFPolicy",
	postprocess_fn=xa_postprocess_nstep_and_prio,
	loss_fn=tf_xasac_actor_critic_loss,
)
XASACTorchPolicy = SACTorchPolicy.with_updates(
	name="XASACTorchPolicy",
	postprocess_fn=xa_postprocess_nstep_and_prio,
	loss_fn=torch_xasac_actor_critic_loss,
)

class XASACTrainer(SACTrainer):
	def get_default_config(cls):
		return XASAC_DEFAULT_CONFIG
		
	@staticmethod
	def execution_plan(workers, config, **kwargs):
		return xadqn_execution_plan(workers, config, **kwargs)
		
	def get_default_policy_class(self, config):
		return XASACTorchPolicy if config["framework"] == "torch" else XASACTFPolicy

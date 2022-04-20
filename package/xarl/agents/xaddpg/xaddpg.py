"""
XADDPG - eXplanation-Aware Deep Deterministic Policy Gradient
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-deterministic-policy-gradients-ddpg-td3
"""  # noqa: E501

from xarl.agents.xadqn import xa_postprocess_nstep_and_prio, xadqn_execution_plan, XADQN_EXTRA_OPTIONS
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_DEFAULT_CONFIG
from ray.rllib.agents.ddpg.td3 import TD3Trainer, TD3_DEFAULT_CONFIG
from ray.rllib.agents.ddpg.ddpg_tf_policy import DDPGTFPolicy
from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy
from xarl.agents.xaddpg.xaddpg_tf_loss import xaddpg_actor_critic_loss as tf_xaddpg_actor_critic_loss
from xarl.agents.xaddpg.xaddpg_torch_loss import xaddpg_actor_critic_loss as torch_xaddpg_actor_critic_loss

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
# XADDPG's Policy
########################

XADDPGTFPolicy = DDPGTFPolicy.with_updates(
	name="XADDPGTFPolicy",
	postprocess_fn=xa_postprocess_nstep_and_prio,
	loss_fn=tf_xaddpg_actor_critic_loss,
)
XADDPGTorchPolicy = DDPGTorchPolicy.with_updates(
	name="XADDPGTorchPolicy",
	postprocess_fn=xa_postprocess_nstep_and_prio,
	loss_fn=torch_xaddpg_actor_critic_loss,
)

class XADDPGTrainer(DDPGTrainer):
	def get_default_config(cls):
		return XADDPG_DEFAULT_CONFIG
		
	@staticmethod
	def execution_plan(workers, config, **kwargs):
		return xadqn_execution_plan(workers, config, **kwargs)
		
	def get_default_policy_class(self, config):
		return XADDPGTorchPolicy if config["framework"] == "torch" else XADDPGTFPolicy

class XATD3Trainer(TD3Trainer):
	def get_default_config(cls):
		return XATD3_DEFAULT_CONFIG
		
	@staticmethod
	def execution_plan(workers, config, **kwargs):
		return xadqn_execution_plan(workers, config, **kwargs)
		
	def get_default_policy_class(self, config):
		return XADDPGTorchPolicy if config["framework"] == "torch" else XADDPGTFPolicy

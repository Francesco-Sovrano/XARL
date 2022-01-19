from ray.rllib.agents.dqn.dqn import GenericOffPolicyTrainer
from ray.rllib.agents.qmix.qmix_policy import QMixTorchPolicy

from ray.rllib.agents.qmix.qmix import QMixTrainer, DEFAULT_CONFIG as QMIX_DEFAULT_CONFIG
from xarl.agents.xadqn import xa_postprocess_nstep_and_prio, xadqn_execution_plan, XADQN_EXTRA_OPTIONS
from xarl.agents.xaqmix.xaqmix_torch_loss import xaddpg_actor_critic_loss as torch_xaddpg_actor_critic_loss

XAQMIX_DEFAULT_CONFIG = QMixTrainer.merge_trainer_configs(
    QMIX_DEFAULT_CONFIG, # For more details, see here: https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
    XADQN_EXTRA_OPTIONS,
    _allow_unknown_configs=True
)


XAQMixTorchPolicy = QMixTorchPolicy.with_updates(
    name="XAQMixTorchPolicy",
    postprocess_fn=xa_postprocess_nstep_and_prio,
    loss_fn=tf_xaddpg_actor_critic_loss,
)

XAQMixTrainer = GenericOffPolicyTrainer.with_updates(
    name="XAQMIX",
    default_config=XAQMIX_DEFAULT_CONFIG,
    default_policy=XAQMixTorchPolicy,
    get_policy_class=None,
    execution_plan=lambda workers, config: xadqn_execution_plan(workers, config, True))

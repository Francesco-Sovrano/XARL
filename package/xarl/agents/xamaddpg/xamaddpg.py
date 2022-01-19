"""Contributed port of MADDPG from OpenAI baselines.

The implementation has a couple assumptions:
- The number of agents is fixed and known upfront.
- Each agent is bound to a policy of the same name.
- Discrete actions are sent as logits (pre-softmax).

For a minimal example, see rllib/examples/two_step_game.py,
and the README for how to run with the multi-agent particle envs.
"""

import logging
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, DEFAULT_POLICY_ID

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


from xarl.agents.xadqn import XADQNTrainer, XADQN_EXTRA_OPTIONS
from xarl.agents.xamaddpg.xamaddpg_policy import XAMADDPGTFPolicy
from ray.rllib.contrib.maddpg import DEFAULT_CONFIG as MADDPG_DEFAULT_CONFIG

from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS

XAMADDPG_DEFAULT_CONFIG = XADQNTrainer.merge_trainer_configs(
	MADDPG_DEFAULT_CONFIG, # For more details, see here: https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
	XADQN_EXTRA_OPTIONS,
	_allow_unknown_configs=True
)
XAMADDPG_DEFAULT_CONFIG["replay_integral_multi_agent_batches"] = True

def before_learn_on_batch(multi_agent_batch, policies, train_batch_size):
	samples = {}

	# Modify keys.
	for pid, p in policies.items():
		i = p.config["agent_id"]
		# print(list(multi_agent_batch.policy_batches.keys()))
		# print(list(policies.keys()))

		keys = multi_agent_batch.policy_batches[pid].data.keys()
		keys = ["_".join([k, str(i)]) for k in keys]
		samples.update(
			dict(
				zip(keys,
					multi_agent_batch.policy_batches[pid].data.values())))

	# Make ops and feed_dict to get "new_obs" from target action sampler.
	new_obs_ph_n = [p.new_obs_ph for p in policies.values()]
	new_obs_n = list()
	for k, v in samples.items():
		if "new_obs" in k:
			new_obs_n.append(v)

	target_act_sampler_n = [p.target_act_sampler for p in policies.values()]
	feed_dict = dict(zip(new_obs_ph_n, new_obs_n))

	new_act_n = p.sess.run(target_act_sampler_n, feed_dict)
	samples.update(
		{"new_actions_%d" % i: new_act
		 for i, new_act in enumerate(new_act_n)})

	# print(list(samples.keys()))
	# Share samples among agents.
	policy_batches = {pid: SampleBatch(samples) for pid in policies.keys()}
	for i,batch in enumerate(policy_batches.values()):
		batch[PRIO_WEIGHTS] = batch[PRIO_WEIGHTS+f'_{i}']
		batch[SampleBatch.INFOS] = batch[SampleBatch.INFOS+f'_{i}']
	return MultiAgentBatch(policy_batches, train_batch_size)

def add_maddpg_postprocessing(config):
	"""Add the before learn on batch hook.

	This hook is called explicitly prior to TrainOneStep() in the execution
	setups for DQN and APEX.
	"""

	def f(batch, workers, config):
		policies = dict(workers.local_worker()
						.foreach_trainable_policy(lambda p, i: (i, p)))
		return before_learn_on_batch(batch, policies,
									 config["train_batch_size"])

	assert config["replay_integral_multi_agent_batches"], "MADDPG requires replay_integral_multi_agent_batches==True"
	config["before_learn_on_batch"] = f
	return config

XAMADDPGTrainer = XADQNTrainer.with_updates(
	name="XAMADDPG",
	default_config=XAMADDPG_DEFAULT_CONFIG,
	default_policy=XAMADDPGTFPolicy,
	get_policy_class=None,
	validate_config=add_maddpg_postprocessing
)


# XAMADDPGTrainer = MADDPGTrainer.with_updates(
#	 name="XAMADDPG",
#	 default_config=XAMADDPG_DEFAULT_CONFIG,
# )

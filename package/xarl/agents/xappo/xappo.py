"""
XAPPO - eXplanation-Aware Asynchronous Proximal Policy Optimization
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#asynchronous-proximal-policy-optimization-appo
"""  # noqa: E501

from more_itertools import unique_everseen

from ray.rllib.agents.impala.impala import *
from ray.rllib.execution.common import (
	STEPS_TRAINED_COUNTER,
	STEPS_TRAINED_THIS_ITER_COUNTER,
	_get_global_vars,
	_get_shared_metrics,
)
from ray.rllib.agents.ppo.appo import *
from ray.rllib.agents.ppo.appo_tf_policy import *
from ray.rllib.agents.ppo.ppo_tf_policy import vf_preds_fetches
from ray.rllib.agents.ppo.appo_torch_policy import AsyncPPOTorchPolicy
# from ray.rllib.evaluation.postprocessing import discount_cumsum
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.policy.view_requirement import ViewRequirement

from xarl.experience_buffers.replay_ops import MixInReplay, get_clustered_replay_buffer, assign_types, add_policy_signature, get_update_replayed_batch_fn, add_buffer_metrics
from xarl.utils.misc import accumulate
from xarl.agents.xappo.xappo_tf_policy import XAPPOTFPolicy
from xarl.agents.xappo.xappo_torch_policy import XAPPOTorchPolicy
from xarl.experience_buffers.replay_buffer import get_batch_infos, get_batch_uid
import random
import numpy as np

XAPPO_EXTRA_OPTIONS = {
	# "lambda": .95, # GAE(lambda) parameter. Taking lambda < 1 introduces bias only when the value function is inaccurate.
	# "batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	# "vtrace": False, # Formula for computing the advantages: batch_mode==complete_episodes implies vtrace==False, thus gae==True.
	##########################################
	"rollout_fragment_length": 2**3, # Number of transitions per batch in the experience buffer
	"train_batch_size": 2**9, # Number of transitions per train-batch
	"replay_proportion": 4, # Set a p>0 to enable experience replay. Saved samples will be replayed with a p:1 proportion to new data samples.
	"gae_with_vtrace": False, # Useful when default "vtrace" is not active. Formula for computing the advantages: it combines GAE with V-Trace.
	"prioritized_replay": True, # Whether to replay batches with the highest priority/importance/relevance for the agent.
	"update_advantages_when_replaying": True, # Whether to recompute advantages when updating priorities.
	"learning_starts": 1, # How many batches to sample before learning starts. Every batch has size 'rollout_fragment_length' (default is 50).
	##########################################
	"buffer_options": {
		'priority_id': 'gains', # Which batch column to use for prioritisation. One of the following: gains, advantages, rewards, prev_rewards, action_logp.
		'priority_lower_limit': None, # A value lower than the lowest possible priority. It depends on the priority_id. By default in DQN and DDPG it is td_error 0, while in PPO it is gain None.
		'priority_aggregation_fn': 'np.mean', # A reduction that takes as input a list of numbers and returns a number representing a batch priority.
		'cluster_size': None, # Default None, implying being equal to global_size. Maximum number of batches stored in a cluster (whose number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'global_size': 2**12, # Default 50000. Maximum number of batches stored in all clusters (whose number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'prioritization_alpha': 0.6, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'prioritization_importance_beta': 0, # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		'prioritization_importance_eta': 1e-2, # Used only if priority_lower_limit is None. A value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0 if beta is > 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		'prioritization_epsilon': 1e-6, # prioritization_epsilon to add to a priority so that it is never equal to 0.
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer instead of the one having the lowest timestamp. In DQN default is 0.
		'stationarity_window_size_for_real_distribution_matching': None, # Whether to use a random number rather than the batch priority during prioritised dropping. If equal to float('inf') then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far. If lower than float('inf') and greater than 0, then the stationarity_window_size_for_real_distribution_matching W is used to guarantee that every W training-steps the buffer is emptied from old state transitions.
		'cluster_prioritisation_strategy': 'sum', # Whether to select which cluster to replay in a prioritised fashion -- Options: None; 'sum', 'avg', 'weighted_avg'.
		'cluster_prioritization_alpha': 1, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'cluster_level_weighting': False, # Whether to use only cluster-level information to compute importance weights rather than the whole buffer.
		'clustering_xi': 4, # Let X be the minimum cluster's size, and C be the number of clusters, and q be clustering_xi, then the cluster's size is guaranteed to be in [X, X+(q-1)CX], with q >= 1, when all clusters have reached the minimum capacity X. This shall help having a buffer reflecting the real distribution of tasks (where each task is associated to a cluster), thus avoiding over-estimation of task's priority.
		# 'clip_cluster_priority_by_max_capacity': False, # Default is False. Whether to clip the clusters priority so that the 'cluster_prioritisation_strategy' will not consider more elements than the maximum cluster capacity. In fact, until al the clusters have reached the minimum size, some clusters may have more elements than the maximum size, to avoid shrinking the buffer capacity with clusters having not enough transitions (i.e. 1 transition).
		'max_age_window': None, # Consider only batches with a relative age within this age window, the younger is a batch the higher will be its importance. Set to None for no age weighting. # Idea from: Fedus, William, et al. "Revisiting fundamentals of experience replay." International Conference on Machine Learning. PMLR, 2020.
	},
	"clustering_scheme": ['Who','How_Well','Why','Where','What','How_Many'], # Which scheme to use for building clusters. Set it to None or to a list of the following: How_WellOnZero, How_Well, When_DuringTraining, When_DuringEpisode, Why, Why_Verbose, Where, What, How_Many, Who
	"clustering_scheme_options": {
		"n_clusters": {
			"who": 4,
			# "why": 8,
			# "what": 8,
		},
		"default_n_clusters": 8,
		"frequency_independent_clustering": False, # Setting this to True can be memory expensive, especially for who explanations
		"agent_action_sliding_window": 2**4,
		"episode_window_size": 2**6, 
		"batch_window_size": 2**8, 
		"training_step_window_size": 2**2,
	},
	"cluster_selection_policy": "min", # Which policy to follow when clustering_scheme is not "none" and multiple explanatory labels are associated to a batch. One of the following: 'random_uniform_after_filling', 'random_uniform', 'random_max', 'max', 'min', 'none'
	"cluster_with_episode_type": False, # Useful with sparse-reward environments. Whether to cluster experience using information at episode-level.
	"cluster_overview_size": 1, # cluster_overview_size <= train_batch_size. If None, then cluster_overview_size is automatically set to train_batch_size. -- When building a single train batch, do not sample a new cluster before x batches are sampled from it. The closer cluster_overview_size is to train_batch_size, the faster is the batch sampling procedure.
	"collect_cluster_metrics": False, # Whether to collect metrics about the experience clusters. It consumes more resources.
	"ratio_of_samples_from_unclustered_buffer": 0, # 0 for no, 1 for full. Whether to sample in a randomised fashion from both a non-prioritised buffer of most recent elements and the XA prioritised buffer.
	"centralised_buffer": True, # for MARL
	"replay_integral_multi_agent_batches": False, # for MARL, set this to True for MADDPG and QMIX
	"batch_dropout_rate": 0, # Probability of dropping a state transition before adding it to the experience buffer. Set this to any value greater than zero to randomly drop state transitions
	# "add_nonstationarity_correction": True, # Experience replay in MARL may suffer from non-stationarity. To avoid this issue a solution is to condition each agent’s value function on a fingerprint that disambiguates the age of the data sampled from the replay memory. To stabilise experience replay, it should be sufficient if each agent’s observations disambiguate where along this trajectory the current training sample originated from. # cit. [2017]Stabilising Experience Replay for Deep Multi-Agent Reinforcement Learning
}
# The combination of update_insertion_time_when_sampling==True and prioritized_drop_probability==0 helps mantaining in the buffer only those batches with the most up-to-date priorities.
XAPPO_DEFAULT_CONFIG = APPOTrainer.merge_trainer_configs(
	DEFAULT_CONFIG, # For more details, see here: https://docs.ray.io/en/master/rllib-algorithms.html#asynchronous-proximal-policy-optimization-appo
	XAPPO_EXTRA_OPTIONS,
	_allow_unknown_configs=True
)

########################
# XAPPO's Execution Plan
########################

class XAPPOTrainer(APPOTrainer):
	def get_default_config(cls):
		return XAPPO_DEFAULT_CONFIG

	def get_default_policy_class(self, config):
		return XAPPOTorchPolicy if config["framework"] == "torch" else XAPPOTFPolicy

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
		random.seed(config["seed"])
		np.random.seed(config["seed"])
		local_replay_buffer, clustering_scheme = get_clustered_replay_buffer(config)
		rollouts = ParallelRollouts(workers, mode="async", num_async=config["max_sample_requests_in_flight_per_worker"])
		local_worker = workers.local_worker()
		
		def add_view_requirements(w):
			for policy in w.policy_map.values():
				# policy.view_requirements[SampleBatch.T] = ViewRequirement(SampleBatch.T, shift=0)
				policy.view_requirements[SampleBatch.INFOS] = ViewRequirement(SampleBatch.INFOS, shift=0)
				policy.view_requirements[SampleBatch.ACTION_LOGP] = ViewRequirement(SampleBatch.ACTION_LOGP, shift=0)
				policy.view_requirements[SampleBatch.NEXT_OBS] = ViewRequirement(SampleBatch.OBS, shift=1)
				policy.view_requirements[SampleBatch.VF_PREDS] = ViewRequirement(SampleBatch.VF_PREDS, shift=0)
				policy.view_requirements[Postprocessing.ADVANTAGES] = ViewRequirement(Postprocessing.ADVANTAGES, shift=0)
				policy.view_requirements[Postprocessing.VALUE_TARGETS] = ViewRequirement(Postprocessing.VALUE_TARGETS, shift=0)
				policy.view_requirements["action_importance_ratio"] = ViewRequirement("action_importance_ratio", shift=0)
				policy.view_requirements["gains"] = ViewRequirement("gains", shift=0)
				if policy.config["buffer_options"]["prioritization_importance_beta"]:
					policy.view_requirements["weights"] = ViewRequirement("weights", shift=0)
				if policy.config["model"]["custom_model_config"].get("add_nonstationarity_correction", False):
					policy.view_requirements["policy_signature"] = ViewRequirement("policy_signature", used_for_compute_actions=True, shift=0)
		workers.foreach_worker(add_view_requirements)

		# Augment with replay and concat to desired train batch size.
		train_batches = rollouts \
			.for_each(lambda batch: batch.decompress_if_needed()) \
			.for_each(lambda batch: assign_types(batch, clustering_scheme, config["rollout_fragment_length"], with_episode_type=config["cluster_with_episode_type"], training_step=local_replay_buffer.get_train_steps())) \
			.flatten() \
			.for_each(MixInReplay(
				local_buffer=local_replay_buffer,
				replay_proportion=config["replay_proportion"],
				cluster_overview_size=config["cluster_overview_size"],
				# update_replayed_fn=get_update_replayed_batch_fn(local_replay_buffer, local_worker, xappo_postprocess_trajectory) if not config['vtrace'] else lambda x:x,
				update_replayed_fn=get_update_replayed_batch_fn(local_replay_buffer, local_worker, xappo_postprocess_trajectory),
				seed=config["seed"],
			)) \
			.flatten() \
			.combine(
				ConcatBatches(
					min_batch_size=config["train_batch_size"],
					count_steps_by=config["multiagent"]["count_steps_by"],
				)
			)

		# Start the learner thread.
		learner_thread = make_learner_thread(local_worker, config)
		learner_thread.start()

		# This sub-flow sends experiences to the learner.
		enqueue_op = train_batches.for_each(Enqueue(learner_thread.inqueue)) 
		# Only need to update workers if there are remote workers.
		if workers.remote_workers():
			enqueue_op = enqueue_op.zip_with_source_actor() \
				.for_each(BroadcastUpdateLearnerWeights(learner_thread, workers, broadcast_interval=config["broadcast_interval"]))

		def increase_train_steps(x):
			local_replay_buffer.increase_train_steps()
			return x

		def record_steps_trained(item):
			count, fetches = item
			metrics = _get_shared_metrics()
			# Manually update the steps trained counter since the learner
			# thread is executing outside the pipeline.
			metrics.counters[STEPS_TRAINED_THIS_ITER_COUNTER] = count
			metrics.counters[STEPS_TRAINED_COUNTER] += count
			return item

		dequeue_op = Dequeue(learner_thread.outqueue, check=learner_thread.is_alive) \
			.for_each(increase_train_steps) \
			.for_each(record_steps_trained)

		merged_op = Concurrently([enqueue_op, dequeue_op], mode="async", output_indexes=[1])

		# Callback for APPO to use to update KL, target network periodically.
		# The input to the callback is the learner fetches dict.
		if config["after_train_step"]:
			merged_op = merged_op \
				.for_each(lambda t: t[1]) \
				.for_each(config["after_train_step"](workers, config))

		standard_metrics_reporting = StandardMetricsReporting(merged_op, workers, config).for_each(learner_thread.add_learner_metrics)
		if config['collect_cluster_metrics']:
			standard_metrics_reporting = standard_metrics_reporting.for_each(lambda x: add_buffer_metrics(x,local_replay_buffer))
		return standard_metrics_reporting
		
"""
XADQN - eXplanation-Aware Deep Q-Networks (DQN, Rainbow, Parametric DQN)
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
"""  # noqa: E501
from more_itertools import unique_everseen
from ray.rllib.agents.dqn.dqn import calculate_rr_weights, DQNTrainer, Concurrently, StandardMetricsReporting, LEARNER_STATS_KEY, DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy, compute_q_values as torch_compute_q_values, torch, F, FLOAT_MIN, TargetNetworkMixin as TorchTargetNetworkMixin, LearningRateSchedule as TorchLearningRateSchedule, build_q_losses as torch_build_q_losses
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy, compute_q_values as tf_compute_q_values, tf, TargetNetworkMixin as TFTargetNetworkMixin, LearningRateSchedule as TFLearningRateSchedule, build_q_losses as tf_build_q_losses
from ray.rllib.evaluation.postprocessing import adjust_nstep
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.execution.train_ops import TrainOneStep, UpdateTargetNetwork, TrainTFMultiGPU
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.execution.common import (
	AGENT_STEPS_SAMPLED_COUNTER,
	STEPS_SAMPLED_COUNTER,
	SAMPLE_TIMER,
	GRAD_WAIT_TIMER,
	_check_sample_batch_type,
	_get_shared_metrics,
)

from deer.experience_buffers.replay_ops import Replay, get_clustered_replay_buffer, assign_types, add_policy_signature, add_buffer_metrics, clean_batch
from deer.experience_buffers.replay_buffer import get_batch_infos, get_batch_uid
from deer.agents.xadqn.xadqn_tf_policy import XADQNTFPolicy
from deer.agents.xadqn.xadqn_torch_policy import XADQNTorchPolicy

import random
import numpy as np

XADQN_EXTRA_OPTIONS = {
	# "rollout_fragment_length": 2**6, # Divide episodes into fragments of this many steps each during rollouts.
	# "train_batch_size": 2**8, # Number of transitions per train-batch
	"learning_starts": 2**14, # How many batches to sample before learning starts. Every batch has size 'rollout_fragment_length' (default is 50).
	"prioritized_replay": True, # Whether to replay batches with the highest priority/importance/relevance for the agent.
	# "batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	##########################################
	"buffer_options": {
		#### ER
		'priority_id': 'td_errors', # Which batch column to use for prioritisation. Default is inherited by DQN and it is 'td_errors'. One of the following: rewards, prev_rewards, td_errors.
		'priority_lower_limit': 0, # A value lower than the lowest possible priority. It depends on the priority_id. By default in DQN and DDPG it is td_error 0, while in PPO it is gain None.
		'priority_aggregation_fn': 'np.mean', # A reduction that takes as input a list of numbers and returns a number representing a batch priority.
		'global_size': 2**14, # Default 50000. Maximum number of batches stored in all clusters (whose number depends on the clustering scheme) of the experience buffer. Every batch has size 'n_step' (default is 1).
		#### PER
		'prioritization_alpha': 0.6, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'prioritization_importance_beta': 0.4, # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		'prioritization_importance_eta': 1e-2, # Used only if priority_lower_limit is None. A value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0 if beta is > 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		'prioritization_epsilon': 1e-6, # prioritization_epsilon to add to a priority so that it is never equal to 0.
		#### XAER
		'cluster_size': None, # Default None, implying being equal to global_size. Maximum number of batches stored in a cluster (whose number depends on the clustering scheme) of the experience buffer. Every batch has size 'n_step' (default is 1).
		'cluster_prioritisation_strategy': 'sum', # Whether to select which cluster to replay in a prioritised fashion -- Options: None; 'sum', 'avg', 'weighted_avg'.
		'cluster_prioritization_alpha': 1, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'cluster_level_weighting': False, # Whether to use only cluster-level information to compute importance weights rather than the whole buffer.
		'clustering_xi': 1, # Let X be the minimum cluster's size, and C be the number of clusters, and q be clustering_xi, then the cluster's size is guaranteed to be in [X, X+(q-1)CX], with q >= 1, when all clusters have reached the minimum capacity X. This shall help having a buffer reflecting the real distribution of tasks (where each task is associated to a cluster), thus avoiding over-estimation of task's priority.
		# 'clip_cluster_priority_by_max_capacity': False, # Default is False. Whether to clip the clusters priority so that the 'cluster_prioritisation_strategy' will not consider more elements than the maximum cluster capacity. In fact, until al the clusters have reached the minimum size, some clusters may have more elements than the maximum size, to avoid shrinking the buffer capacity with clusters having not enough transitions (i.e. 1 transition).
		#### DEER
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer instead of the one having the lowest timestamp. In DQN default is 0.
		'global_distribution_matching': False, # Whether to use a random number rather than the batch priority during prioritised dropping. If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far. 
		'stationarity_window_size': None, # If lower than float('inf') and greater than 0, then the stationarity_window_size W is used to guarantee that every W training-steps the buffer is emptied from old state transitions.
		'stationarity_smoothing_factor': 1, # A number >= 1, where 1 means no smoothing. The larger this number, the smoother the transition from a stationarity stage to the next one. This should help avoiding experience buffers saturated by one single episode during a stage transition. The optimal value should be equal to ceil(HORIZON*number_of_agents/EXPERIENCE_BUFFER_SIZE)*stationarity_window_size.
		#### Extra
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
XADQN_DEFAULT_CONFIG = DQNTrainer.merge_trainer_configs(
	DQN_DEFAULT_CONFIG, # For more details, see here: https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
	XADQN_EXTRA_OPTIONS,
	_allow_unknown_configs=True
)

########################
# XADQN's Execution Plan
########################	

class XADQNTrainer(DQNTrainer):
	def get_default_config(cls):
		return XADQN_DEFAULT_CONFIG

	def get_default_policy_class(self, config):
		return XADQNTorchPolicy if config["framework"] == "torch" else XADQNTFPolicy

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
		replay_batch_size = config["train_batch_size"]
		sample_batch_size = config.get("n_step",1)
		if sample_batch_size and sample_batch_size > 1:
			replay_batch_size = int(max(1, replay_batch_size // sample_batch_size))
		local_replay_buffer, clustering_scheme = get_clustered_replay_buffer(config)
		local_worker = workers.local_worker()

		def add_view_requirements(w):
			for policy in w.policy_map.values():
				# policy.view_requirements[SampleBatch.T] = ViewRequirement(SampleBatch.T, shift=0)
				policy.view_requirements[SampleBatch.INFOS] = ViewRequirement(SampleBatch.INFOS, shift=0)
				if policy.config["buffer_options"]["priority_id"] == "td_errors":
					policy.view_requirements["td_errors"] = ViewRequirement("td_errors", shift=0)
				if policy.config["model"]["custom_model_config"].get("add_nonstationarity_correction", False):
					policy.view_requirements["policy_signature"] = ViewRequirement("policy_signature", used_for_compute_actions=True, shift=0)
		workers.foreach_worker(add_view_requirements)

		rollouts = ParallelRollouts(workers, mode="bulk_sync")
		
		# We execute the following steps concurrently:
		# (1) Generate rollouts and store them in our local replay buffer. Calling
		# next() on store_op drives this.
		def store_batch(batch):
			sub_batch_iter = assign_types(batch, clustering_scheme, sample_batch_size, with_episode_type=config["cluster_with_episode_type"], training_step=local_replay_buffer.get_train_steps())
			total_buffer_additions = sum(map(local_replay_buffer.add_batch, sub_batch_iter))
			return batch
		store_op = rollouts.for_each(store_batch)

		# (2) Read and train on experiences from the replay buffer. Every batch
		# returned from the LocalReplay() iterator is passed to TrainOneStep to
		# take a SGD step, and then we decide whether to update the target network.
		def update_priorities(item):
			local_replay_buffer.increase_train_steps()
			samples, info_dict = item
			if not config.get("prioritized_replay"):
				return info_dict
			priority_id = config["buffer_options"]["priority_id"]
			if priority_id == "td_errors":
				for policy_id, info in info_dict.items():
					td_errors = info.get("td_error", info[LEARNER_STATS_KEY].get("td_error"))
					# samples.policy_batches[policy_id].set_get_interceptor(None)
					samples.policy_batches[policy_id]["td_errors"] = td_errors
			# IMPORTANT: split train-batch into replay-batches, using batch_uid, before updating priorities
			policy_batch_list = []
			for policy_id, batch in samples.policy_batches.items():
				if sample_batch_size > 1 and config["batch_mode"] == "complete_episodes":
					sub_batch_indexes = [
						i
						for i,infos in enumerate(batch['infos'])
						if "batch_uid" in infos
					] + [batch.count]
					sub_batch_iter = (
						batch.slice(sub_batch_indexes[j], sub_batch_indexes[j+1])
						for j in range(len(sub_batch_indexes)-1)
					)
				else:
					sub_batch_iter = batch.timeslices(sample_batch_size)
				sub_batch_iter = unique_everseen(sub_batch_iter, key=get_batch_uid)
				for i,sub_batch in enumerate(sub_batch_iter):
					if i >= len(policy_batch_list):
						policy_batch_list.append({})
					policy_batch_list[i][policy_id] = sub_batch
			for policy_batch in policy_batch_list:
				local_replay_buffer.update_priorities(policy_batch)
			return info_dict
		post_fn = config.get("before_learn_on_batch") or (lambda b, *a: b)
		if config.get("simple_optimizer",True):
			train_step_op = TrainOneStep(workers)
		else:
			train_step_op = TrainTFMultiGPU(
				workers=workers,
				sgd_minibatch_size=config["train_batch_size"],
				num_sgd_iter=1,
				num_gpus=config["num_gpus"],
				# shuffle_sequences=True,
				_fake_gpus=config["_fake_gpus"],
				# framework=config.get("framework"),
			)
		concat_batch_dict = {
			'min_batch_size': replay_batch_size,
			'count_steps_by': config["multiagent"]["count_steps_by"]
		}
		replay_op = Replay(
				local_buffer=local_replay_buffer, 
				replay_batch_size=replay_batch_size, 
				cluster_overview_size=config["cluster_overview_size"]
			) \
			.flatten() \
			.combine(ConcatBatches(**concat_batch_dict)) \
			.for_each(lambda x: post_fn(x, workers, config)) \
			.for_each(train_step_op) \
			.for_each(update_priorities) \
			.for_each(UpdateTargetNetwork(workers, config["target_network_update_freq"]))

		# Alternate deterministically between (1) and (2). Only return the output
		# of (2) since training metrics are not available until (2) runs.
		train_op = Concurrently([store_op, replay_op], mode="round_robin", output_indexes=[1], round_robin_weights=calculate_rr_weights(config))

		standard_metrics_reporting = StandardMetricsReporting(train_op, workers, config)
		if config['collect_cluster_metrics']:
			standard_metrics_reporting = standard_metrics_reporting.for_each(lambda x: add_buffer_metrics(x,local_replay_buffer))
		return standard_metrics_reporting
		
	
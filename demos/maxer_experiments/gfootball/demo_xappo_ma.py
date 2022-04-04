# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray
from ray.tune.registry import get_trainable_cls, _global_registry, ENV_CREATOR
import time
from xarl.utils.workflow import train

from xarl.agents.xappo import XAPPOTrainer, XAPPO_DEFAULT_CONFIG
from environments import *
from xarl.models.appo import TFAdaptiveMultiHeadNet
from ray.rllib.models import ModelCatalog
from xarl.models.head_generator.adaptive_model_wrapper import get_tf_heads_model, get_heads_input
# Register the models to use.
ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet.init(get_tf_heads_model, get_heads_input))

SELECT_ENV = "gfootball"

CENTRALISED_TRAINING = True
NUM_AGENTS = 3

CONFIG = XAPPO_DEFAULT_CONFIG.copy()
CONFIG["env_config"] = {
	'num_agents': NUM_AGENTS,
	'number_of_left_players_agent_controls': NUM_AGENTS,
	'env_name': 'test_example_multiagent', 
	'stacked': False,
	'logdir': os.environ["TUNE_RESULT_DIR"],
	'write_goal_dumps': False, 
	'write_full_episode_dumps': False, 
	'render': False,
	'dump_frequency': 0,
	'channel_dimensions': (42, 42)
}
CONFIG.update({
	"horizon": 256, # Number of steps after which the episode is forced to terminate. Defaults to `env.spec.max_episode_steps` (if present) for Gym envs.
	"model": { # this is for GraphDrive and GridDrive
		"vf_share_layers": True, # Share layers for value function. If you set this to True, it's important to tune vf_loss_coeff.
		"custom_model": "adaptive_multihead_network",
	},
	# "preprocessor_pref": "rllib", # this prevents reward clipping on Atari and other weird issues when running from checkpoints
	"gamma": 0.999, # We use an higher gamma to extend the MDP's horizon; optimal agency on GraphDrive requires a longer horizon.
	"seed": 42, # This makes experiments reproducible.
	"rollout_fragment_length": 2**3, # Number of transitions per batch in the experience buffer. Default is 50 for APPO.
	"train_batch_size": 2**7, # Number of transitions per train-batch. Default is: 100 for TD3, 256 for SAC and DDPG, 32 for DQN, 500 for APPO.
	"replay_proportion": 4, # Set a p>0 to enable experience replay. Saved samples will be replayed with a p:1 proportion to new data samples.
	"replay_buffer_num_slots": 2**9, # Maximum number of batches stored in the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).	
	###################################
	# 'lambda': 0.95,
	# 'kl_coeff': 0.2,
	# 'clip_rewards': False,
	# 'vf_clip_param': 10.0,
	# 'entropy_coeff': 0.01,
	# 'train_batch_size': 2000,
	# 'sample_batch_size': 100,
	# 'sgd_minibatch_size': 500,
	# 'num_sgd_iter': 10,
	# 'num_workers': 10,
	# 'num_envs_per_worker': 1,
	# 'batch_mode': 'truncate_episodes',
	# 'observation_filter': 'NoFilter',
	# 'vf_share_layers': 'true',
	# 'num_gpus': 1,
	# 'lr': 2.5e-4,
	# 'log_level': 'DEBUG',
	# 'simple_optimizer': True,
	###################################
	"gae_with_vtrace": False, # Useful when default "vtrace" is not active. Formula for computing the advantages: it combines GAE with V-Trace.
	"prioritized_replay": True, # Whether to replay batches with the highest priority/importance/relevance for the agent.
	"update_advantages_when_replaying": True, # Whether to recompute advantages when updating priorities.
	# "learning_starts": 2**12, # How many steps of the model to sample before learning starts. Every batch has size 'rollout_fragment_length' (default is 50).
	'vtrace': True,
	##################################
	"buffer_options": {
		'priority_id': 'gains', # Which batch column to use for prioritisation. One of the following: gains, advantages, rewards, prev_rewards, action_logp.
		'priority_lower_limit': None, # A value lower than the lowest possible priority. It depends on the priority_id. By default in DQN and DDPG it is td_error 0, while in PPO it is gain None.
		'priority_aggregation_fn': 'np.mean', # A reduction that takes as input a list of numbers and returns a number representing a batch priority.
		'cluster_size': None, # Default None, implying being equal to global_size. Maximum number of batches stored in a cluster (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'global_size': 2**12, # Default 50000. Maximum number of batches stored in all clusters (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'prioritization_alpha': 0.6, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'prioritization_importance_beta': 0, # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		'prioritization_importance_eta': 1e-2, # Used only if priority_lower_limit is None. A value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0 if beta is > 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		'prioritization_epsilon': 1e-6, # prioritization_epsilon to add to a priority so that it is never equal to 0.
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer instead of the one having the lowest timestamp. In DQN default is 0.
		'global_distribution_matching': False, # Whether to use a random number rather than the batch priority during prioritised dropping. If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far.
		'cluster_prioritisation_strategy': 'sum', # Whether to select which cluster to replay in a prioritised fashion -- Options: None; 'sum', 'avg', 'weighted_avg'.
		'cluster_prioritization_alpha': 1, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'cluster_level_weighting': True, # Whether to use only cluster-level information to compute importance weights rather than the whole buffer.
		'clustering_xi': 4, # Let X be the minimum cluster's size, and C be the number of clusters, and q be clustering_xi, then the cluster's size is guaranteed to be in [X, X+(q-1)CX], with q >= 1, when all clusters have reached the minimum capacity X. This shall help having a buffer reflecting the real distribution of tasks (where each task is associated to a cluster), thus avoiding over-estimation of task's priority.
		# 'clip_cluster_priority_by_max_capacity': False, # Whether to clip the clusters priority so that the 'cluster_prioritisation_strategy' will not consider more elements than the maximum cluster capacity.
		'max_age_window': None, # Consider only batches with a relative age within this age window, the younger is a batch the higher will be its importance. Set to None for no age weighting. # Idea from: Fedus, William, et al. "Revisiting fundamentals of experience replay." International Conference on Machine Learning. PMLR, 2020.
	},
	"clustering_scheme": [ # Which scheme to use for building clusters. Set it to None or to a list of the following: How_WellOnZero, How_Well, When_DuringTraining, When_DuringEpisode, Why, Why_Verbose, Where, What, How_Many, Who
		'Who',
		'How_Well',
		'Why',
		# 'Where',
		# 'What',
		# 'How_Many'
	],
	"clustering_scheme_options": {
		"n_clusters": {
			"who": 4,
			# "why": 8,
			# "what": 8,
		},
		"default_n_clusters": 8,
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
	# 'batch_mode': 'complete_episodes',
})
CONFIG["callbacks"] = CustomEnvironmentCallbacks
# framework = CONFIG.get("framework","tf")
# if framework in ["tf2", "tf", "tfe"]:
# 	from ray.rllib.models.tf.fcnet import FullyConnectedNetwork as FCNet, Keras_FullyConnectedNetwork as Keras_FCNet
# elif framework == "torch":
# 	from ray.rllib.models.torch.fcnet import (FullyConnectedNetwork as FCNet)
# ModelCatalog.register_custom_model("fcnet", FCNet)

# Setup MARL training strategy: centralised or decentralised
env = _global_registry.get(ENV_CREATOR, SELECT_ENV)(CONFIG["env_config"])
obs_space = env.observation_space
act_space = env.action_space
def gen_policy():
	return (None, obs_space, act_space, {})
policy_graphs = {}
if not CENTRALISED_TRAINING:
	for i in range(NUM_AGENTS):
		policy_graphs[f'agent-{i}'] = gen_policy()
	def policy_mapping_fn(agent_id):
			return f'agent-{agent_id}'
else:
	policy_graphs['centralised_agent'] = gen_policy()
	def policy_mapping_fn(agent_id):
			return 'centralised_agent'

CONFIG.update({
	"multiagent": {
		"policies": policy_graphs,
		"policy_mapping_fn": policy_mapping_fn,
		# Which metric to use as the "batch size" when building a
		# MultiAgentBatch. The two supported values are:
		# env_steps: Count each time the env is "stepped" (no matter how many
		#   multi-agent actions are passed/how many multi-agent observations
		#   have been returned in the previous step).
		# agent_steps: Count each individual agent step as one step.
		# "count_steps_by": "env_steps",
	},
})

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True, include_dashboard=False)

train(XAPPOTrainer, CONFIG, SELECT_ENV, test_every_n_step=1e7, stop_training_after_n_step=4e7)
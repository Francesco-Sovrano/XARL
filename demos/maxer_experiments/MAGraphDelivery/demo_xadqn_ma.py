# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
# os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
import multiprocessing
import json
import shutil
import ray
from ray.tune.registry import get_trainable_cls, _global_registry, ENV_CREATOR
import time
from xarl.utils.workflow import train
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from xarl.agents.xadqn import XADQNTrainer, XADQN_DEFAULT_CONFIG
from environments import *

SELECT_ENV = "MAGraphDelivery-FullWorldSomeAgents"
CENTRALISED_TRAINING = True
NUM_AGENTS = 16
VISIBILITY_RADIUS = 10

CONFIG = XADQN_DEFAULT_CONFIG.copy()
CONFIG["env_config"] = {
	'num_agents': NUM_AGENTS,
	'n_discrete_actions': 36,
	'reward_fn': 'unitary_more_frequent', # one of the following: 'frequent', 'more_frequent', 'sparse', 'unitary_frequent', 'unitary_more_frequent', 'unitary_sparse'
	'fairness_type_fn': 'simple', # one of the following: None, 'simple', 'engineered'
	'fairness_reward_fn': 'simple', # one of the following: None, 'simple', 'engineered', 'unitary_engineered'
	'visibility_radius': VISIBILITY_RADIUS,
	'spawn_on_sources_only': True,
	'max_refills_per_source': float('inf'),
	'max_deliveries_per_target': 1,
	'target_junctions_number': NUM_AGENTS//4,
	'source_junctions_number': 1,
	################################
	'max_dimension': 32,
	'junctions_number': 32,
	'max_roads_per_junction': 4,
	'junction_radius': 1,
	'max_distance_to_path': .5, # meters
	################################
	'random_seconds_per_step': False, # whether to sample seconds_per_step from an exponential distribution
	'mean_seconds_per_step': 1, # in average, a step every n seconds
	################################
	'min_speed': 0.2, # m/s
	'max_speed': 2, # m/s
}
CONFIG.update({
	"framework": "torch",
	"horizon": 2**9, # Number of steps after which the episode is forced to terminate. Defaults to `env.spec.max_episode_steps` (if present) for Gym envs.
	# "no_done_at_end": True, # IMPORTANT: this allows lifelong learning with decent bootstrapping
	"model": { # this is for GraphDrive and GridDrive
		# "vf_share_layers": True, # Share layers for value function. If you set this to True, it's important to tune vf_loss_coeff.
		"custom_model": "comm_adaptive_multihead_network",
		"custom_model_config": {
			"comm_range": VISIBILITY_RADIUS,
			# 'max_num_neighbors': 32,
			# 'message_size': 128,
		},
	},
	# "normalize_actions": False,

	# "model": { # this is for GraphDrive and GridDrive
	# 	"vf_share_layers": True, # Share layers for value function. If you set this to True, it's important to tune vf_loss_coeff.
	# 	"custom_model": "adaptive_multihead_network",
	# },
	# "preprocessor_pref": "rllib", # this prevents reward clipping on Atari and other weird issues when running from checkpoints
	"gamma": 0.999, # We use an higher gamma to extend the MDP's horizon; optimal agency on GraphDrive requires a longer horizon.
	"seed": 42, # This makes experiments reproducible.
	###########################
	"rollout_fragment_length": 2**6, # Divide episodes into fragments of this many steps each during rollouts. Default is 1.
	"train_batch_size": 2**8, # Number of transitions per train-batch. Default is: 100 for TD3, 256 for SAC and DDPG, 32 for DQN, 500 for APPO.
	# "batch_mode": "truncate_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	###########################
	"prioritized_replay": True, # Whether to replay batches with the highest priority/importance/relevance for the agent.
	'buffer_size': 2**14, # Size of the experience buffer. Default 50000
	"learning_starts": 2**14, # How many steps of the model to sample before learning starts.
	"prioritized_replay_alpha": 0.6,
	"prioritized_replay_beta": 0.4, # The smaller this is, the stronger is over-sampling
	"prioritized_replay_eps": 1e-6,
	###########################
	# 'lr': .0000625,
	# 'adam_epsilon': .00015,
	# 'exploration_config': {
	# 	'epsilon_timesteps': 200000,
	# 	'final_epsilon': 0.01,
	# },
	# 'timesteps_per_iteration': 10000,
	###########################
	# "grad_clip": None, # no need of gradient clipping with huber loss
	# "dueling": True,
	# "double_q": True,
	# "num_atoms": 21,
	# "v_max": 2**5,
	# "v_min": -1,
	"clip_rewards": False,
	##################################
	"buffer_options": {
		'priority_id': 'td_errors', # Which batch column to use for prioritisation. Default is inherited by DQN and it is 'td_errors'. One of the following: rewards, prev_rewards, td_errors.
		'priority_lower_limit': 0, # A value lower than the lowest possible priority. It depends on the priority_id. By default in DQN and DDPG it is td_error 0, while in PPO it is gain None.
		'priority_aggregation_fn': 'np.mean', # A reduction that takes as input a list of numbers and returns a number representing a batch priority.
		'cluster_size': None, # Default None, implying being equal to global_size. Maximum number of batches stored in a cluster (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'global_size': 2**15, # Default 50000. Maximum number of batches stored in all clusters (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'prioritization_alpha': 0.6, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'prioritization_importance_beta': 0.4, # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		'prioritization_importance_eta': 1e-2, # Used only if priority_lower_limit is None. A value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0 if beta is > 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		'prioritization_epsilon': 1e-6, # prioritization_epsilon to add to a priority so that it is never equal to 0.
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer instead of the one having the lowest timestamp. In DQN default is 0.
		'global_distribution_matching': False, # Whether to use a random number rather than the batch priority during prioritised dropping. If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far.
		'cluster_prioritisation_strategy': 'sum', # Whether to select which cluster to replay in a prioritised fashion -- Options: None; 'sum', 'avg', 'weighted_avg'.
		'cluster_prioritization_alpha': 1, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'cluster_level_weighting': True, # Whether to use only cluster-level information to compute importance weights rather than the whole buffer.
		'clustering_xi': 1, # Let X be the minimum cluster's size, and C be the number of clusters, and q be clustering_xi, then the cluster's size is guaranteed to be in [X, X+(q-1)CX], with q >= 1, when all clusters have reached the minimum capacity X. This shall help having a buffer reflecting the real distribution of tasks (where each task is associated to a cluster), thus avoiding over-estimation of task's priority.
		# 'clip_cluster_priority_by_max_capacity': False, # Whether to clip the clusters priority so that the 'cluster_prioritisation_strategy' will not consider more elements than the maximum cluster capacity.
		'max_age_window': None, # Consider only batches with a relative age within this age window, the younger is a batch the higher will be its importance. Set to None for no age weighting. # Idea from: Fedus, William, et al. "Revisiting fundamentals of experience replay." International Conference on Machine Learning. PMLR, 2020.
	},
	"clustering_scheme": [ # Which scheme to use for building clusters. Set it to None or to a list of the following: How_WellOnZero, How_Well, When_DuringTraining, When_DuringEpisode, Why, Why_Verbose, Where, What, How_Many, Who
		'Who',
		'How_Well',
		'How_Fair',
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
})
CONFIG["callbacks"] = CustomEnvironmentCallbacks

# Register models
from ray.rllib.models import ModelCatalog
from xarl.models import get_model_catalog_dict
for k,v in get_model_catalog_dict('dqn', CONFIG["framework"]).items():
	ModelCatalog.register_custom_model(k, v)

# Setup MARL training strategy: centralised or decentralised
env = _global_registry.get(ENV_CREATOR, SELECT_ENV)(CONFIG["env_config"])
obs_space = env.observation_space
act_space = env.action_space
if not CENTRALISED_TRAINING:
	policy_graphs = {
		f'agent-{i}': (None, obs_space, act_space, CONFIG) 
		for i in range(NUM_AGENTS)
	}
	policy_mapping_fn = lambda agent_id: f'agent-{agent_id}'
else:
	# policy_graphs = {DEFAULT_POLICY_ID: (None, obs_space, act_space, CONFIG)}
	policy_graphs = {}
	policy_mapping_fn = lambda agent_id: DEFAULT_POLICY_ID

CONFIG["centralised_buffer"] = CENTRALISED_TRAINING
CONFIG["multiagent"].update({
	"policies": policy_graphs,
	"policy_mapping_fn": policy_mapping_fn,
	# Optional list of policies to train, or None for all policies.
	"policies_to_train": None,
	# Optional function that can be used to enhance the local agent
	# observations to include more state.
	# See rllib/evaluation/observation_function.py for more info.
	"observation_fn": None,
	# When replay_mode=lockstep, RLlib will replay all the agent
	# transitions at a particular timestep together in a batch. This allows
	# the policy to implement differentiable shared computations between
	# agents it controls at that timestep. When replay_mode=independent,
	# transitions are replayed independently per policy.
	"replay_mode": "independent", # XAER does not support "lockstep", yet
	# Which metric to use as the "batch size" when building a
	# MultiAgentBatch. The two supported values are:
	# env_steps: Count each time the env is "stepped" (no matter how many
	#   multi-agent actions are passed/how many multi-agent observations
	#   have been returned in the previous step).
	# agent_steps: Count each individual agent step as one step.
	"count_steps_by": "agent_steps", # XAER does not support "env_steps"?
})
print('Config:', CONFIG)

####################################################################################
####################################################################################

ray.shutdown()
ray.init(
	ignore_reinit_error=True, 
	include_dashboard=False, 
	log_to_driver=False, 
	num_cpus=os.cpu_count(),
)

train(XADQNTrainer, CONFIG, SELECT_ENV, test_every_n_step=4e7//100, stop_training_after_n_step=4e7)
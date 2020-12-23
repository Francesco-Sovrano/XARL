# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray

from xarl.agents.xaddpg import XADDPGTrainer, XADDPG_DEFAULT_CONFIG
from environments import *

# SELECT_ENV = "ToyExample-v0"
# SELECT_ENV = "CescoDrive-v2"
SELECT_ENV = "AlexDrive-v0"

CONFIG = XADDPG_DEFAULT_CONFIG.copy()
CONFIG["log_level"] = "WARN"
# For more config options, see here: https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
CONFIG["prioritized_replay"] = True
CONFIG["filter_duplicated_batches_when_replaying"] = False # Whether to remove duplicated batches from a replay batch (n.b. the batch size will remain the same, new unique batches will be sampled until the expected size is reached).
CONFIG["buffer_options"] = {
	'priority_id': "weights", # What batch column to use for prioritisation. One of the following: rewards, prev_rewards, weights
	'priority_aggregation_fn': 'lambda x: np.mean(np.abs(x))', # A reduce function that takes as input a list of numbers and returns a number representing a batch's priority
	'cluster_size': 50000, # Default 50000. Maximum number of batches stored in a cluster (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
	'global_size': 50000, # Default 50000. Maximum number of batches stored in all clusters (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
	'alpha': 0.6, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
	'beta': 0.4, # Parameter that regulates a mechanism for computing importance sampling.
	'epsilon': 1e-6, # Epsilon to add to a priority so that it is never equal to 0.
	'prioritized_drop_probability': 0.5, # Probability of dropping experience with the lowest priority in the buffer
	'global_distribution_matching': False, # "If True, then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that at any given time the sampled experiences will approximately match the distribution of all samples seen so far."
	'prioritised_cluster_sampling': True, # Whether to select which cluster to replay in a prioritised fashion
	'sample_simplest_unknown_task': 'above_average', # Whether to sample the simplest unknown task with higher probability. Two options: 'average': the one with the cluster priority closest to the average cluster priority; 'above_average': the one with the cluster priority closest to the cluster with the smallest priority greater than the average cluster priority. It requires prioritised_cluster_sampling==True.
}
CONFIG["clustering_scheme"] = "moving_best_extrinsic_reward_with_type" # Which scheme to use for building clusters. One of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type, reward_with_multiple_types, moving_best_extrinsic_reward_with_multiple_types
CONFIG["batch_mode"] = "complete_episodes" # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes' otherwise it can also be 'truncate_episodes'
CONFIG["update_only_sampled_cluster"] = True # Whether to update the priority only in the sampled cluster and not in all, if the same batch is in more than one cluster. Setting this option to True causes a slighlty higher memory consumption but shall increase by far the speed in updating priorities.

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True)

# Configure RLlib to train a policy using the “Taxi-v3” environment and a PPO optimizer
agent = XADDPGTrainer(CONFIG, env=SELECT_ENV)

# Inspect the trained policy and model, to see the results of training in detail
# policy = agent.get_policy()
# model = policy.model
# print(model.base_model.summary())

# Train a policy. The following code runs 30 iterations and that’s generally enough to begin to see improvements in the “Taxi-v3” problem
# results = []
# episode_data = []
# episode_json = []
n = 0
while True:
	n += 1
	result = agent.train()
	# print(result)
	# results.append(result)
	episode = {
		'n': n, 
		'episode_reward_min': result['episode_reward_min'], 
		'episode_reward_mean': result['episode_reward_mean'], 
		'episode_reward_max': result['episode_reward_max'],  
		'episode_len_mean': result['episode_len_mean']
	}
	# episode_data.append(episode)
	# episode_json.append(json.dumps(episode))
	# file_name = agent.save(checkpoint_root)
	print(f'{n+1:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}, len mean: {result["episode_len_mean"]:8.4f}, train ratio: {(result["info"]["num_steps_trained"]/result["info"]["num_steps_sampled"]):8.4f}')
	# print(f'Checkpoint saved to {file_name}')


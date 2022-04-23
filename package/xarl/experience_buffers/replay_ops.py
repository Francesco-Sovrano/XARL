from typing import List
import random
import numpy as np
from more_itertools import unique_everseen

from ray.util.iter import LocalIterator, _NextValueNotReady
from ray.util.iter_metrics import SharedMetrics
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, DEFAULT_POLICY_ID

from xarl.experience_buffers.replay_buffer import SimpleReplayBuffer, LocalReplayBuffer, get_batch_infos
from xarl.experience_buffers.clustering_scheme import *

def get_clustered_replay_buffer(config):
	assert config["batch_mode"] == "complete_episodes" or not config["cluster_with_episode_type"], f"This algorithm requires 'complete_episodes' as batch_mode when 'cluster_with_episode_type' is True"
	clustering_scheme_type = config.get("clustering_scheme", None)
	# no need for unclustered_buffer if clustering_scheme_type is none
	ratio_of_samples_from_unclustered_buffer = config["ratio_of_samples_from_unclustered_buffer"] if clustering_scheme_type else 0
	local_replay_buffer = LocalReplayBuffer(
		prioritized_replay=config["prioritized_replay"],
		buffer_options=config["buffer_options"], 
		learning_starts=config["learning_starts"], 
		seed=config["seed"],
		cluster_selection_policy=config["cluster_selection_policy"],
		ratio_of_samples_from_unclustered_buffer=ratio_of_samples_from_unclustered_buffer,
		centralised_buffer=config["centralised_buffer"],
		replay_integral_multi_agent_batches=config["replay_integral_multi_agent_batches"],
	)
	clustering_scheme = ClusterManager(clustering_scheme_type, config["clustering_scheme_options"])
	return local_replay_buffer, clustering_scheme

def assign_types(multi_batch, clustering_scheme, batch_fragment_length, with_episode_type=True, training_step=None):
	if not isinstance(multi_batch, MultiAgentBatch):
		multi_batch = MultiAgentBatch({DEFAULT_POLICY_ID: multi_batch}, multi_batch.count)
	
	if not with_episode_type:
		batch_list = multi_batch.timeslices(batch_fragment_length) if multi_batch.count > batch_fragment_length else [multi_batch]
		for i,batch in enumerate(batch_list):
			for pid,sub_batch in batch.policy_batches.items():
				get_batch_infos(sub_batch)['batch_type'] = clustering_scheme.get_batch_type(sub_batch, training_step=training_step, episode_step=i, agent_id=pid)		
		return batch_list

	batch_dict = {}
	for pid,b in multi_batch.policy_batches.items():
		batch_dict[pid] = []
		for episode in b.split_by_episode():
			sub_batch_list = episode.as_multi_agent().timeslices(batch_fragment_length) if episode.count > batch_fragment_length else [episode]
			sub_batch_list = list(map(lambda x: x.policy_batches[DEFAULT_POLICY_ID], sub_batch_list))
			episode_type = clustering_scheme.get_episode_type(sub_batch_list)
			for i,sub_batch in enumerate(sub_batch_list):
				get_batch_infos(sub_batch)['batch_type'] = clustering_scheme.get_batch_type(sub_batch, episode_type=episode_type, training_step=training_step, episode_step=i, agent_id=pid)
			batch_dict[pid] += sub_batch_list
	batch_list = [
		MultiAgentBatch({
			pid: b
			for pid,b in zip(batch_dict.keys(),b_list)
		},b_list[0].count)
		for b_list in zip(*batch_dict.values())
	]
	return batch_list

def get_update_replayed_batch_fn(local_replay_buffer, local_worker, postprocess_trajectory_fn):
	def update_replayed_fn(samples):
		if isinstance(samples, MultiAgentBatch):
			for pid, batch in samples.policy_batches.items():
				if pid not in local_worker.policies_to_train:
					continue
				policy = local_worker.get_policy(pid)
				samples.policy_batches[pid] = postprocess_trajectory_fn(policy, batch)
			local_replay_buffer.update_priorities(samples.policy_batches)
		else:
			local_replay_buffer.update_priorities({
				pid:postprocess_trajectory_fn(policy, samples)
				for pid, policy in local_worker.policy_map.items()
			})
		return samples
	return update_replayed_fn

def clean_batch(batch, keys_to_keep=None, keep_only_keys_to_keep=False):
	if isinstance(batch, MultiAgentBatch):
		for b in batch.policy_batches.values():
			for k,v in list(b.items()):
				if keys_to_keep and k in keys_to_keep:
					continue
				if keep_only_keys_to_keep or not isinstance(v, np.ndarray):
					del b[k]
	else:
		for k,v in list(batch.items()):
			if keys_to_keep and k in keys_to_keep:
				continue
			if keep_only_keys_to_keep or not isinstance(v, np.ndarray):
				del batch[k]
	return batch

def add_buffer_metrics(results, buffer):
	results['buffer']=buffer.stats()
	return results

class StoreToReplayBuffer:
	def __init__(self, local_buffer: LocalReplayBuffer = None):
		self.local_actor = local_buffer
		
	def __call__(self, batch: SampleBatchType):
		self.local_actor.add_batch(batch)
		return batch

def Replay(local_buffer, replay_batch_size=1, cluster_overview_size=None, update_replayed_fn=None):
	def gen_replay(_):
		while True:
			batch_list = local_buffer.replay(
				batch_count=replay_batch_size, 
				cluster_overview_size=cluster_overview_size,
				update_replayed_fn=update_replayed_fn,
			)
			if not batch_list:
				yield _NextValueNotReady()
			else:
				yield batch_list
	return LocalIterator(gen_replay, SharedMetrics())

class MixInReplay:
	"""This operator adds replay to a stream of experiences.

	It takes input batches, and returns a list of batches that include replayed
	data as well. The number of replayed batches is determined by the
	configured replay proportion. The max age of a batch is determined by the
	number of replay slots.
	"""

	def __init__(self, local_buffer, replay_proportion, cluster_overview_size=None, update_replayed_fn=None, seed=None):
		random.seed(seed)
		np.random.seed(seed)
		self.replay_buffer = local_buffer
		self.replay_proportion = replay_proportion
		self.update_replayed_fn = update_replayed_fn
		self.cluster_overview_size = cluster_overview_size

	def __call__(self, sample_batch):
		# n = np.random.poisson(self.replay_proportion)
		n = int(self.replay_proportion//1)
		if self.replay_proportion%1 > 0 and random.random() <= self.replay_proportion%1:
			n += 1
		output_batches = []
		# Put sample_batch in the experience buffer and add it to the output_batches
		if not isinstance(sample_batch, MultiAgentBatch):
			sample_batch = MultiAgentBatch({DEFAULT_POLICY_ID: sample_batch}, sample_batch.count)
		output_batches.append(sample_batch)
		self.replay_buffer.add_batch(sample_batch) # Set update_prioritisation_weights=True for updating importance weights
		# Sample n batches from the buffer
		if self.replay_buffer.can_replay() and n > 0:
			output_batches += self.replay_buffer.replay(
				batch_count=n,
				cluster_overview_size=self.cluster_overview_size,
				update_replayed_fn=self.update_replayed_fn,
			)
		return output_batches

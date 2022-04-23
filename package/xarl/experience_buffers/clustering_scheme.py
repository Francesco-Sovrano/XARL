# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter, deque
from more_itertools import unique_everseen
from xarl.utils.running_statistics import RunningStats
from ray.rllib.policy.sample_batch import SampleBatch
import itertools
from sklearn.cluster import *

class none:
	def __init__(self, **args):
		pass

	def get_episode_type(self, episode):
		return 'episode:none'

	def get_batch_type(self, batch, **args):
		return [('explanation:none',)]

class InfoExplanation(none): # Explanations coming from the info_dict generated by the environment at each step
	def __init__(self, info_type, n_clusters=None, default_n_clusters=8, **args):
		if n_clusters is None:
			n_clusters = {}
		self.n_clusters = n_clusters.get(info_type, default_n_clusters)
		# super().__init__(episode_window_size, batch_window_size)
		print(f'[{info_type}] n_clusters={self.n_clusters}')
		self.info_type = info_type
		self.clusterer = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.n_clusters) if self.n_clusters else None # MiniBatchKMeans allows online clustering
		self.explanation_vector_labels = set()

	def get_batch_type(self, batch, **args):
		explanation_iter = map(lambda x: x["explanation"].get(self.info_type,'none') if "explanation" in x else 'none', batch[SampleBatch.INFOS])
		explanation_iter = map(lambda x: x if isinstance(x,(list,tuple)) else [x], explanation_iter)
		explanation_iter = itertools.chain(*explanation_iter)
		explanation_iter = unique_everseen(explanation_iter, key=str)
		explanation_list = list(explanation_iter)

		if self.clusterer:
			explanation_vector_list = list(filter(lambda x: isinstance(x, np.ndarray), explanation_list))
			new_explanation_vector_labels = set(map(np.array2string, explanation_vector_list)) - self.explanation_vector_labels
			if new_explanation_vector_labels:
				self.explanation_vector_labels |= new_explanation_vector_labels
				X = list(filter(lambda x: np.array2string(x) in new_explanation_vector_labels, explanation_vector_list))
				self.clusterer.partial_fit(X*self.n_clusters) # online learning
		explanation_iter = map(lambda x: f'cluster_{self.clusterer.predict([x])[0]}' if isinstance(x, np.ndarray) else x, explanation_list)

		explanation_iter = map(lambda x: (f"{self.info_type}:{x}",), explanation_iter)
		return list(explanation_iter)

class Why(InfoExplanation):
	def __init__(self, **args):
		super().__init__('why', **args)

class Why_Verbose(Why):
	def get_batch_type(self, batch):
		explanation_iter = super().get_batch_type(batch)
		return [tuple(sorted(explanation_iter))]

class Where(InfoExplanation):
	def __init__(self, **args):
		super().__init__('where', **args)

class What(InfoExplanation): # What: information about what is inside the observation (i.e. intersection of type A, B, etc.)
	def __init__(self, **args):
		super().__init__('what', **args)

class How_Many(InfoExplanation): # How Many: how many different agents are seen within an observation.
	def __init__(self, **args):
		super().__init__('how-many', **args)

#### Special Explanations
class How_WellOnZero(none):
	def get_episode_type(self, episode):
		episode_extrinsic_reward = sum((np.sum(batch["rewards"]) for batch in episode))
		# episode_extrinsic_reward = np.sum(episode[-1]["rewards"])
		return 'episode:how:better_than_zero' if episode_extrinsic_reward > 0 else 'episode:how:worse_than_zero' # Best batches = batches that lead to positive extrinsic reward

	def get_batch_type(self, batch, **args):
		batch_extrinsic_reward = np.sum(batch["rewards"])
		batch_type = 'how:better_than_zero' if batch_extrinsic_reward > 0 else 'how:worse_than_zero'
		return [(batch_type,)]

class How_Well(none):
	def __init__(self, episode_window_size=2**6, batch_window_size=2**8, **args):
		print(f'[how] episode_window_size={episode_window_size}, batch_window_size={batch_window_size}')
		self.episode_stats = RunningStats(window_size=episode_window_size)
		self.batch_stats = RunningStats(window_size=batch_window_size)

	def get_episode_type(self, episode):
		episode_extrinsic_reward = sum((np.sum(batch["rewards"]) for batch in episode))
		# episode_extrinsic_reward = np.sum(episode[-1]["rewards"])
		self.episode_stats.push(episode_extrinsic_reward)
		return 'episode:how:better_than_average' if episode_extrinsic_reward > self.episode_stats.mean else 'episode:how:worse_than_average'

	def get_H(self, batch):
		batch_extrinsic_reward = np.sum(batch["rewards"])
		self.batch_stats.push(batch_extrinsic_reward)
		return 'better_than_average' if batch_extrinsic_reward > self.batch_stats.mean else 'worse_than_average'

	def get_batch_type(self, batch, **args):
		return [(f"how:{self.get_H(batch)}",)]

class When_DuringEpisode(none): # When: information about the training step of a batch.
	def __init__(self, **args):
		print(f'[when-episode]')

	def get_batch_type(self, batch, episode_step=None, **args):
		if episode_step is None: 
			episode_step = 'unknown'
		return [(f"when:episode_step_{episode_step}",)]

class When_DuringTraining(none): # When: information about the training step of a batch.
	def __init__(self, training_step_window_size=2**4, **args):
		print(f'[when-training] window_size={training_step_window_size}')
		self.training_step_window_size = training_step_window_size

	def get_batch_type(self, batch, training_step=None, **args):
		if training_step is None: 
			training_step = 'unknown'
		else:
			training_step %= self.training_step_window_size
		return [(f"when:training_step_{training_step}",)]

class Who(InfoExplanation): # Who: a clustering of sequences of actions (and observation embeddings?) taken by a single agent during an episode or part of it.
	def __init__(self, agent_action_sliding_window=16, **args):
		super().__init__('who', **args)
		print(f'[who] agent_action_sliding_window={agent_action_sliding_window}')
		self.agent_action_dict = {}
		self.agent_last_episode_dict = {}
		self.agent_action_sliding_window = agent_action_sliding_window

	def get_who(self, action_deque):
		action_list = list(action_deque)
		action_list += [np.zeros_like(action_list[0])]*(self.agent_action_sliding_window-len(action_list))
		return np.array(action_list).flatten()

	def get_batch_type(self, batch, agent_id='none', **args):
		who_list = []
		last_episode_id = self.agent_last_episode_dict.get(agent_id, None)
		action_deque = self.agent_action_dict.get(agent_id,None)
		if action_deque is None:
			action_deque = self.agent_action_dict[agent_id] = deque(maxlen=self.agent_action_sliding_window)
		for action,episode_id in zip(batch[SampleBatch.ACTIONS], batch[SampleBatch.EPS_ID]):
			if episode_id != last_episode_id:
				if last_episode_id is not None:
					who_list.append(self.get_who(action_deque))
				action_deque.clear()
				self.agent_last_episode_dict[agent_id] = episode_id
			action_deque.append(action)
		who_list.append(self.get_who(action_deque))
		# print(0, who_list)

		for x in batch["infos"]:
			explanation_dict = x.get("explanation",None)
			if explanation_dict is None:
				explanation_dict = x["explanation"] = {}
			explanation_dict[self.info_type] = who_list

		return super().get_batch_type(batch)

#### Manager
class ClusterManager:
	def __init__(self, cluster_type_list, clustering_scheme_options):
		self.cluster_type_list = sorted(unique_everseen(cluster_type_list)) if cluster_type_list else ('none',)
		self.cluster_label_generator_list = [
			eval(cluster_type)(**clustering_scheme_options)
			for cluster_type in self.cluster_type_list
		]
		
	def get_episode_type(self, episode):
		return tuple(
			gen.get_episode_type(episode) 
			for gen in self.cluster_label_generator_list
		)
			
	def get_batch_type(self, batch, episode_type=None, **args):
		if not episode_type:
			episode_type = 'episode:none'
		cluster_type_list = [
			gen.get_batch_type(batch, **args)
			for gen in self.cluster_label_generator_list
		]
		# print(cluster_type_list)
		# merge episode type with batch type
		batch_type_iter = [
			(episode_type,) + t
			for t in cluster_type_list[0]
		]
		# combine batch types
		for cluster_type in cluster_type_list[1:]:
			batch_type_iter = [
				l + r
				for l in batch_type_iter
				for r in cluster_type
			]
		return tuple(batch_type_iter)

# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter
from more_itertools import unique_everseen
from xarl.utils.running_statistics import RunningStats
import itertools
from sklearn.cluster import *

class none():
	def __init__(self, **args):
		pass

	def get_episode_type(self, episode):
		return 'none'

	def get_batch_type(self, batch, episode_type='none'):
		return ((episode_type,'none'),)

class positive_H(none):
	def get_episode_type(self, episode):
		episode_extrinsic_reward = sum((np.sum(batch["rewards"]) for batch in episode))
		# episode_extrinsic_reward = np.sum(episode[-1]["rewards"])
		return 'better' if episode_extrinsic_reward > 0 else 'worse' # Best batches = batches that lead to positive extrinsic reward

	def get_batch_type(self, batch, episode_type='none'):
		batch_extrinsic_reward = np.sum(batch["rewards"])
		batch_type = 'greater' if batch_extrinsic_reward > 0 else 'lower'
		return ((episode_type, batch_type),)

class H(none):
	def __init__(self, episode_window_size=2**6, batch_window_size=2**8, **args):
		print(f'[H] episode_window_size={episode_window_size}, batch_window_size={batch_window_size}')
		self.episode_stats = RunningStats(window_size=episode_window_size)
		self.batch_stats = RunningStats(window_size=batch_window_size)

	def get_episode_type(self, episode):
		episode_extrinsic_reward = sum((np.sum(batch["rewards"]) for batch in episode))
		# episode_extrinsic_reward = np.sum(episode[-1]["rewards"])
		self.episode_stats.push(episode_extrinsic_reward)
		return 'better' if episode_extrinsic_reward > self.episode_stats.mean else 'worse'

	def get_H(self, batch):
		batch_extrinsic_reward = np.sum(batch["rewards"])
		self.batch_stats.push(batch_extrinsic_reward)
		return 'greater' if batch_extrinsic_reward > self.batch_stats.mean else 'lower'

	def get_batch_type(self, batch, episode_type='none'):
		return ((episode_type, self.get_H(batch)),)

class W(H):
	def __init__(self, episode_window_size=2**6, batch_window_size=2**8, n_clusters=8, **args):
		super().__init__(episode_window_size, batch_window_size)
		print(f'[W] episode_window_size={episode_window_size}, batch_window_size={batch_window_size}, n_clusters={n_clusters}')
		self.n_clusters = n_clusters
		self.clusterer = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.n_clusters) if self.n_clusters else None # MiniBatchKMeans allows online clustering
		self.explanation_vector_labels = set()

	def get_batch_type(self, batch, episode_type='none'):
		explanation_iter = map(lambda x: x.get("explanation",'None'), batch["infos"])
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

		explanation_iter = map(lambda x:(episode_type, x), explanation_iter)
		return tuple(explanation_iter)

	# def get_batch_type(self, batch, episode_type='none'):
	# 	explanation_iter = map(lambda x: x.get("explanation",'None'), batch["infos"])
	# 	explanation_iter = map(lambda x: x if isinstance(x,(list,tuple)) else [x], explanation_iter)
	# 	explanation_iter = itertools.chain(*explanation_iter)
	# 	explanation_iter = unique_everseen(explanation_iter)
	# 	explanation_iter = map(lambda x:(episode_type, x), explanation_iter)
	# 	return tuple(explanation_iter)

class long_W(W):
	def get_batch_type(self, batch, episode_type='none'):
		explanation_iter = super().get_batch_type(batch, episode_type)
		return ((episode_type, sorted(explanation_iter)),)
		
class HW(W):
	def get_batch_type(self, batch, episode_type='none'):
		explanation_iter = super().get_batch_type(batch, episode_type)
		batch_type = self.get_H(batch)
		explanation_iter = map(lambda x: (x[0],batch_type,x[1]), explanation_iter)
		return tuple(explanation_iter)

# Who: a clustering of sequences of actions (and observation embeddings?) taken by a single agent during an episode or part of it.
# How Many: how many different agents are seen within an observation.
# When: information about the performed training steps.
# What: information about what is inside the observation (i.e. intersection of type A, B, etc.)

class long_HW(HW):
	def get_batch_type(self, batch, episode_type='none'):
		explanation_iter = super().get_batch_type(batch, episode_type)
		batch_type = explanation_iter[0][-2]
		explanation_iter = map(lambda x:x[-1], explanation_iter)
		return ((episode_type, batch_type, sorted(explanation_iter)),)

# class ClusterManager():

# 	def __init__(self, cluster_typeset, cluster_generator_args):
# 		self.cluster_label_generator_list = [
# 			eval(f'{cluster_type}(**cluster_generator_args)')
# 			for cluster_type in cluster_typeset
# 		]
		
# 	def get_episode_type(self, episode):
# 		return tuple(
# 			gen.get_episode_type(episode) 
# 			for gen in self.cluster_label_generator_list
# 		)
			
# 	def get_batch_type(self, batch, episode_type='none'):
# 		cluster_type_list = [
# 			gen.get_batch_type(batch)
# 			for gen in self.cluster_label_generator_list
# 		]
# 		# merge episode type with batch type
# 		batch_type_iter = (
# 			(episode_type,) + t
# 			for t in cluster_type_list[0]
# 		)
# 		# combine batch types
# 		for cluster_type in cluster_type_list[1:]:
# 			batch_type_iter = (
# 				l + r
# 				for l in batch_type_iter
# 				for r in cluster_type
# 			)
# 		return tuple(batch_type_iter)

# -*- coding: utf-8 -*-
import utils.tensorflow_utils as tf_utils
import tensorflow.compat.v1 as tf
import numpy as np
import itertools as it
from agent.algorithm.loss.policy_loss import PolicyLoss
from agent.algorithm.loss.value_loss import ValueLoss
from utils.distributions import Categorical, Normal
from agent.network import *
from collections import deque
from utils.statistics import Statistics
from agent.algorithm.advantage_estimator import *
#===============================================================================
# from utils.running_std import RunningMeanStd
#===============================================================================
import options
flags = options.get()

def merge_splitted_advantages(advantage):
	return flags.extrinsic_coefficient*advantage[0] + flags.intrinsic_coefficient*advantage[1]

class AC_Algorithm(object):
	extract_importance_weight = flags.advantage_estimator.lower() in ["vtrace","gae_v"]

	@staticmethod
	def get_reversed_cumulative_return(gamma, last_value, reversed_reward, reversed_value, reversed_extra, reversed_importance_weight):
		return eval(flags.advantage_estimator.lower())(
			gamma=gamma, 
			last_value=last_value, 
			reversed_reward=reversed_reward, 
			reversed_value=reversed_value, 
			reversed_extra=reversed_extra, 
			reversed_importance_weight=reversed_importance_weight,
		)
	
	def __init__(self, group_id, model_id, environment_info, beta=None, training=True, parent=None, sibling=None):
		self.parameters_type = eval('tf.{}'.format(flags.parameters_type))
		self.beta = beta if beta is not None else flags.beta
		self.value_count = 2 if flags.split_values else 1
		# initialize
		self.training = training
		self.group_id = group_id
		self.model_id = model_id
		self.id = '{0}_{1}'.format(self.group_id,self.model_id) # model id
		self.parent = parent if parent is not None else self # used for sharing with other models in hierarchy, if any
		self.sibling = sibling if sibling is not None else self # used for sharing with other models in hierarchy, if any
		# Environment info
		action_shape = environment_info['action_shape']
		self.policy_heads = [
			{
				'size':head[0], # number of actions to take
				'depth':head[1] if len(head) > 1 else 0 # number of discrete action types: set 0 for continuous control
			}
			for head in action_shape
		]
		state_shape = environment_info['state_shape']
		self.state_heads = [
			{'shape':head}
			for head in state_shape
		]
		self.state_scaler = environment_info['state_scaler'] # state scaler, for saving memory (eg. in case of RGB input: uint8 takes less memory than float64)
		self.has_masked_actions = environment_info['has_masked_actions']
		# Create the network
		self.build_input_placeholders()
		self.initialize_network()
		self.build_network()
		# Stuff for building the big-batch and optimize training computations
		self._big_batch_feed = [{},{}]
		self._batch_count = [0,0]
		self._train_batch_size = flags.batch_size*flags.big_batch_size
		# Statistics
		self._train_statistics = Statistics(flags.episode_count_for_evaluation)
		#=======================================================================
		# self.loss_distribution_estimator = RunningMeanStd(batch_size=flags.batch_size)
		#=======================================================================
		self.actor_loss_is_too_small = False
		
	def get_statistics(self):
		return self._train_statistics.get()
	
	def build_input_placeholders(self):
		print( "Building network {} input placeholders".format(self.id) )
		self.constrain_replay = flags.constraining_replay and flags.replay_mean > 0
		self.is_replayed_batch = self._scalar_placeholder(dtype=tf.bool, batch_size=1, name="replay")
		self.state_mean_batch = [self._state_placeholder(shape=head['shape'], batch_size=1, name="state_mean{}".format(i)) for i,head in enumerate(self.state_heads)] 
		self.state_std_batch = [self._state_placeholder(shape=head['shape'], batch_size=1, name="state_std{}".format(i)) for i,head in enumerate(self.state_heads)]
		self.state_batch = [self._state_placeholder(shape=head['shape'], name="state{}".format(i)) for i,head in enumerate(self.state_heads)]
		self.new_state_batch = [self._state_placeholder(shape=head['shape'], name="new_state{}".format(i)) for i,head in enumerate(self.state_heads)]
		self.size_batch = self._scalar_placeholder(dtype=tf.int32, name="size")
		for i,state in enumerate(self.state_batch):
			print( "	[{}]State{} shape: {}".format(self.id, i, state.get_shape()) )
		for i,state in enumerate(self.new_state_batch):
			print( "	[{}]New State{} shape: {}".format(self.id, i, state.get_shape()) )
		self.reward_batch = self._value_placeholder("reward")
		print( "	[{}]Reward shape: {}".format(self.id, self.reward_batch.get_shape()) )
		self.cumulative_return_batch = self._value_placeholder("cumulative_return")
		print( "	[{}]Cumulative Return shape: {}".format(self.id, self.cumulative_return_batch.get_shape()) )
		self.advantage_batch = self._value_placeholder("advantage")
		print( "	[{}]Advantage shape: {}".format(self.id, self.advantage_batch.get_shape()) )
		self.old_state_value_batch = self._value_placeholder("old_state_value")
		self.old_policy_batch = [self._policy_placeholder(policy_size=head['size'], policy_depth=head['depth'], name="old_policy{}".format(i)) for i,head in enumerate(self.policy_heads)]
		self.old_action_batch = [self._action_placeholder(policy_size=head['size'], policy_depth=head['depth'], name="old_action_batch{}".format(i)) for i,head in enumerate(self.policy_heads)]
		if self.has_masked_actions:
			self.old_action_mask_batch = [self._action_placeholder(policy_size=head['size'], policy_depth=1, name="old_action_mask_batch{}".format(i)) for i,head in enumerate(self.policy_heads)]
			
	def _policy_placeholder(self, policy_size, policy_depth, name=None, batch_size=None):
		if is_continuous_control(policy_depth):
			shape = [batch_size,2,policy_size]
		else: # Discrete control
			shape = [batch_size,policy_size,policy_depth] if policy_size > 1 else [batch_size,policy_depth]
		return tf.placeholder(dtype=self.parameters_type, shape=shape, name=name)
			
	def _action_placeholder(self, policy_size, policy_depth, name=None, batch_size=None):
		shape = [batch_size]
		if policy_size > 1 or is_continuous_control(policy_depth):
			shape.append(policy_size)
		if policy_depth > 1:
			shape.append(policy_depth)
		return tf.placeholder(dtype=self.parameters_type, shape=shape, name=name)

	def _shaped_placeholder(self, name=None, shape=None, dtype=None):
		if dtype is None:
			dtype=self.parameters_type
		return tf.placeholder(dtype=dtype, shape=shape, name=name)
		
	def _value_placeholder(self, name=None, batch_size=None, dtype=None):
		return self._shaped_placeholder(name=name, shape=[batch_size,self.value_count], dtype=dtype)
	
	def _scalar_placeholder(self, name=None, batch_size=None, dtype=None):
		return self._shaped_placeholder(name=name, shape=[batch_size], dtype=dtype)
		
	def _state_placeholder(self, shape, name=None, batch_size=None):
		shape = [batch_size] + list(shape)
		input = tf.zeros(shape if batch_size is not None else [1] + shape[1:], dtype=self.parameters_type) # default value
		return tf.placeholder_with_default(input=input, shape=shape, name=name) # with default we can use batch normalization directly on it
		
	def build_optimizer(self, optimization_algoritmh):
		# global step
		global_step = tf.Variable(0, trainable=False)
		# learning rate
		learning_rate = tf_utils.get_annealable_variable(
			function_name=flags.alpha_annealing_function, 
			initial_value=flags.alpha, 
			global_step=global_step, 
			decay_steps=flags.alpha_decay_steps, 
			decay_rate=flags.alpha_decay_rate
		) if flags.alpha_decay else flags.alpha
		# gradient optimizer
		optimizer = {}
		for p in self.get_network_partitions():
			optimizer[p] = tf_utils.get_optimization_function(optimization_algoritmh)(learning_rate=learning_rate, use_locking=True)
		print("Gradient {} optimized by {}".format(self.id, optimization_algoritmh))
		return optimizer, global_step
	
	def get_network_partitions(self):
		return ['Actor','Critic','Reward','TransitionPredictor']
	
	def initialize_network(self, qvalue_estimation=False):
		self.network = {}
		batch_dict = {
			'state': self.state_batch, 
			'state_mean': self.state_mean_batch,
			'state_std': self.state_std_batch,
			'new_state': self.new_state_batch, 
			'size': self.size_batch,
			'action': self.old_action_batch,
			'reward': self.reward_batch,
		}
		# Build intrinsic reward network here because we need its internal state for building actor and critic
		self.network['Reward'] = IntrinsicReward_Network(id=self.id, batch_dict=batch_dict, scope_dict={'self': "IRNet{0}".format(self.id)}, training=self.training)
		if flags.intrinsic_reward:
			reward_network_output = self.network['Reward'].build()
			self.intrinsic_reward_batch = reward_network_output[0]
			self.intrinsic_reward_loss = reward_network_output[1]
			self.training_state = reward_network_output[2]
			print( "	[{}]Intrinsic Reward shape: {}".format(self.id, self.intrinsic_reward_batch.get_shape()) )
			print( "	[{}]Training State Kernel shape: {}".format(self.id, self.training_state['kernel'].get_shape()) )
			print( "	[{}]Training State Bias shape: {}".format(self.id, self.training_state['bias'].get_shape()) )		
			batch_dict['training_state'] = self.training_state
		# Build actor and critic
		for p in ['Actor','Critic','TransitionPredictor']:
			if flags.separate_actor_from_critic: # non-shared graph
				node_id = self.id + p
				parent_id = self.parent.id + p
				sibling_id = self.sibling.id + p
			else: # shared graph
				node_id = self.id
				parent_id = self.parent.id
				sibling_id = self.sibling.id
			scope_dict = {
				'self': "Net{0}".format(node_id),
				'parent': "Net{0}".format(parent_id),
				'sibling': "Net{0}".format(sibling_id)
			}
			self.network[p] = eval('{}_Network'.format(flags.network_configuration))(
				id=node_id, 
				qvalue_estimation=qvalue_estimation,
				policy_heads=self.policy_heads,
				batch_dict=batch_dict,
				scope_dict=scope_dict, 
				training=self.training,
				value_count=self.value_count,
				state_scaler=self.state_scaler
			)
				
	def build_network(self):
		# Actor & Critic
		self.network['Actor'].build(name='Actor', has_actor=True, has_critic=False, has_transition_predictor=False, use_internal_state=flags.network_has_internal_state)
		self.actor_batch = self.network['Actor'].policy_batch
		for i,b in enumerate(self.actor_batch): 
			print( "	[{}]Actor{} output shape: {}".format(self.id, i, b.get_shape()) )
		self.network['Critic'].build(name='Critic', has_actor=False, has_critic=True, has_transition_predictor=False, use_internal_state=flags.network_has_internal_state)
		self.critic_batch = self.network['Critic'].value_batch
		print( "	[{}]Critic output shape: {}".format(self.id, self.critic_batch.get_shape()) )
		# Sample action, after getting keys
		self.action_batch, self.hot_action_batch = self.sample_actions()
		for i,b in enumerate(self.action_batch): 
			print( "	[{}]Action{} output shape: {}".format(self.id, i, b.get_shape()) )
		for i,b in enumerate(self.hot_action_batch): 
			print( "	[{}]HotAction{} output shape: {}".format(self.id, i, b.get_shape()) )
		if flags.with_transition_predictor:
			self.network['TransitionPredictor'].build(name='TransitionPredictor', has_actor=False, has_critic=False, has_transition_predictor=True, use_internal_state=flags.network_has_internal_state)
			self.relevance_batch = self.network['TransitionPredictor'].relevance_batch
			self.new_transition_prediction_batch = self.network['TransitionPredictor'].new_transition_prediction_batch
			self.new_state_embedding_batch = self.network['TransitionPredictor'].new_state_embedding_batch
			self.reward_prediction_batch = self.network['TransitionPredictor'].reward_prediction_batch
			print( "	[{}]Relevance shape: {}".format(self.id, self.relevance_batch.get_shape()) )
			
	def sample_actions(self):
		action_batch = []
		hot_action_batch = []
		for h,actor_head in enumerate(self.actor_batch):
			if is_continuous_control(self.policy_heads[h]['depth']):
				new_policy_batch = tf.transpose(actor_head, [1, 0, 2])
				sample_batch = Normal(new_policy_batch[0], new_policy_batch[1]).sample()
				action = tf.clip_by_value(sample_batch, -1,1, name='action_clipper')
				action_batch.append(action) # Sample action batch in forward direction, use old action in backward direction
				hot_action_batch.append(action)
			else: # discrete control
				distribution = Categorical(actor_head)
				action = distribution.sample(one_hot=False) # Sample action batch in forward direction, use old action in backward direction
				action_batch.append(action)
				hot_action_batch.append(distribution.get_sample_one_hot(action))
		# Give self esplicative name to output for easily retrieving it in frozen graph
		# tf.identity(action_batch, name="action")
		return action_batch, hot_action_batch
		
	def prepare_loss(self, global_step):
		self.global_step = global_step
		print( "Preparing loss {}".format(self.id) )
		self.state_value_batch = self.critic_batch
		# [Actor loss]
		policy_builder = PolicyLoss(
			global_step= self.global_step,
			type= flags.policy_loss,
			beta= self.beta,
			policy_heads= self.policy_heads, 
			actor_batch= self.actor_batch,
			old_policy_batch= self.old_policy_batch, 
			old_action_batch= self.old_action_batch, 
			is_replayed_batch= self.is_replayed_batch,
			old_action_mask_batch= self.old_action_mask_batch if self.has_masked_actions else None,
		)
		# if flags.runtime_advantage:
		# 	self.advantage_batch = adv = self.cumulative_return_batch - self.state_value_batch # baseline is always up to date
		self.policy_loss = policy_builder.get(tf.map_fn(fn=merge_splitted_advantages, elems=self.advantage_batch) if self.value_count > 1 else self.advantage_batch)
		self.importance_weight_batch = policy_builder.get_importance_weight_batch()
		print( "	[{}]Importance Weight shape: {}".format(self.id, self.importance_weight_batch.get_shape()) )
		self.policy_kl_divergence = policy_builder.approximate_kullback_leibler_divergence()
		self.policy_clipping_frequency = policy_builder.get_clipping_frequency()
		self.policy_entropy_regularization = policy_builder.get_entropy_regularization()
		# [Critic loss]
		self.value_loss = flags.value_coefficient * ValueLoss(
			global_step=self.global_step,
			loss=flags.value_loss,
			prediction=self.state_value_batch, 
			old_prediction=self.old_state_value_batch, 
			target=self.cumulative_return_batch
		).get()

		self.extra_loss = 0
		# [Entropy regularization]
		if not flags.intrinsic_reward and flags.entropy_regularization:
			self.extra_loss += -self.policy_entropy_regularization
		# [Constraining Replay]
		if self.constrain_replay:
			constrain_loss = sum(
				0.5*builder.reduce_function(tf.squared_difference(new_distribution.mean(), tf.stop_gradient(old_action))) 
				for builder, new_distribution, old_action in zip(policy_loss_builder, new_policy_distributions, self.old_action_batch)
			)
			self.extra_loss += tf.cond(
				pred=self.is_replayed_batch[0], 
				true_fn=lambda: constrain_loss,
				false_fn=lambda: tf.constant(0., dtype=self.parameters_type)
			)
		# [RND loss]
		if flags.intrinsic_reward:
			self.extra_loss += self.intrinsic_reward_loss
		# [State Predictor loss]
		if flags.with_transition_predictor:
			self.transition_predictor_loss = flags.transition_predictor_coefficient * ValueLoss(
				global_step=self.global_step,
				loss='vanilla',
				prediction=self.new_transition_prediction_batch, 
				target=self.new_state_embedding_batch
			).get() + ValueLoss(
				global_step=self.global_step,
				loss='vanilla',
				prediction=self.reward_prediction_batch, 
				target=self.reward_batch
			).get()
			self.extra_loss += self.transition_predictor_loss
		# [Total loss]
		self.total_loss = self.policy_loss + self.value_loss + self.extra_loss
		
	def get_shared_keys(self, partitions=None):
		if partitions is None:
			partitions = self.get_network_partitions()
		# set removes duplicates
		key_list = set(it.chain.from_iterable(self.network[p].shared_keys for p in partitions))
		return sorted(key_list, key=lambda x: x.name)
	
	def get_update_keys(self, partitions=None):
		if partitions is None:
			partitions = self.get_network_partitions()
		# set removes duplicates
		key_list = set(it.chain.from_iterable(self.network[p].update_keys for p in partitions))
		return sorted(key_list, key=lambda x: x.name)

	def _get_train_op(self, global_step, optimizer, loss, shared_keys, update_keys, global_keys):
		with tf.control_dependencies(update_keys): # control_dependencies is for batch normalization
			grads_and_vars = optimizer.compute_gradients(loss=loss, var_list=shared_keys)
			# grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
			grad, vars = zip(*grads_and_vars)
			global_grads_and_vars = tuple(zip(grad, global_keys))
			return optimizer.apply_gradients(global_grads_and_vars, global_step=global_step)
		
	def minimize_local_loss(self, optimizer, global_step, global_agent): # minimize loss and apply gradients to global vars.
		actor_optimizer, critic_optimizer, reward_optimizer, transition_predictor_optimizer = optimizer.values()
		self.actor_op = self._get_train_op(
			global_step=global_step,
			optimizer=actor_optimizer, 
			loss=self.policy_loss, 
			shared_keys=self.get_shared_keys(['Actor']), 
			global_keys=global_agent.get_shared_keys(['Actor']),
			update_keys=self.get_update_keys(['Actor'])
		)
		self.critic_op = self._get_train_op(
			global_step=global_step,
			optimizer=critic_optimizer, 
			loss=self.value_loss, 
			shared_keys=self.get_shared_keys(['Critic']), 
			global_keys=global_agent.get_shared_keys(['Critic']),
			update_keys=self.get_update_keys(['Critic'])
		)
		if flags.intrinsic_reward:
			self.reward_op = self._get_train_op(
				global_step=global_step,
				optimizer=reward_optimizer, 
				loss=self.intrinsic_reward_loss, 
				shared_keys=self.get_shared_keys(['Reward']), 
				global_keys=global_agent.get_shared_keys(['Reward']),
				update_keys=self.get_update_keys(['Reward'])
			)
		if flags.with_transition_predictor:
			self.transition_predictor_op = self._get_train_op(
				global_step=global_step,
				optimizer=transition_predictor_optimizer, 
				loss=self.transition_predictor_loss, 
				shared_keys=self.get_shared_keys(['TransitionPredictor']), 
				global_keys=global_agent.get_shared_keys(['TransitionPredictor']),
				update_keys=self.get_update_keys(['TransitionPredictor'])
			)
			
	def bind_sync(self, src_network, name=None):
		with tf.name_scope(name, "Sync{0}".format(self.id),[]) as name:
			src_vars = src_network.get_shared_keys()
			dst_vars = self.get_shared_keys()
			sync_ops = []
			for(src_var, dst_var) in zip(src_vars, dst_vars):
				sync_op = tf.assign(dst_var, src_var) # no need for locking dst_var
				sync_ops.append(sync_op)
			self.sync_op = tf.group(*sync_ops, name=name)
				
	def sync(self):
		tf.get_default_session().run(fetches=self.sync_op)
		
	def predict_reward(self, info_dict):
		assert flags.intrinsic_reward, "Cannot get intrinsic reward if the RND layer is not built"
		# State
		feed_dict = self._get_multihead_feed(target=self.new_state_batch, source=info_dict['new_states'])
		feed_dict.update( self._get_multihead_feed(target=self.state_mean_batch, source=[info_dict['state_mean']]) )
		feed_dict.update( self._get_multihead_feed(target=self.state_std_batch, source=[info_dict['state_std']]) )
		# Return intrinsic_reward
		return tf.get_default_session().run(fetches=self.intrinsic_reward_batch, feed_dict=feed_dict)

	def predict_transition_relevance(self, info_dict):
		assert flags.with_transition_predictor, "Cannot get transition relevance if the state predictor layer is not built"
		# State
		feed_dict = self._get_multihead_feed(target=self.state_batch, source=info_dict['states'])
		feed_dict.update( self._get_multihead_feed(target=self.old_action_batch, source=info_dict['actions']) )
		feed_dict.update( self._get_multihead_feed(target=self.new_state_batch, source=info_dict['new_states']) )
		feed_dict.update( {self.reward_batch: info_dict['rewards']} )
		# Return relevance
		return tf.get_default_session().run(fetches=self.relevance_batch, feed_dict=feed_dict)
				
	def predict_value(self, info_dict):
		state_batch = info_dict['states']
		size_batch = info_dict['sizes']
		bootstrap = info_dict['bootstrap']
		for i,b in enumerate(bootstrap):
			state_batch = state_batch + [b['state']]
			size_batch[i] += 1
		# State
		feed_dict = self._get_multihead_feed(target=self.state_batch, source=state_batch)
		# Internal State
		if flags.network_has_internal_state:
			feed_dict.update( self._get_internal_state_feed(info_dict['internal_states']) )
			feed_dict.update( {self.size_batch: size_batch} )
		# Return value_batch
		value_batch = tf.get_default_session().run(fetches=self.state_value_batch, feed_dict=feed_dict)
		return value_batch[:-1], value_batch[-1], None
	
	def predict_action(self, info_dict):
		batch_size = info_dict['sizes']
		batch_count = len(batch_size)
		# State
		feed_dict = self._get_multihead_feed(target=self.state_batch, source=info_dict['states'])
		# Internal state
		if flags.network_has_internal_state:
			feed_dict.update( self._get_internal_state_feed( info_dict['internal_states'] ) )
			feed_dict.update( {self.size_batch: batch_size} )
		# Return action_batch, policy_batch, new_internal_state
		action_batch, hot_action_batch, policy_batch, value_batch, new_internal_states = tf.get_default_session().run(
			fetches=[
				self.action_batch, 
				self.hot_action_batch, 
				self.actor_batch, 
				self.state_value_batch, 
				self._get_internal_state(),
			], 
			feed_dict=feed_dict
		)
		# Properly format for output the internal state
		if len(new_internal_states) == 0:
			new_internal_states = [new_internal_states]*batch_count
		else:
			new_internal_states = [
				[
					[
						sub_partition_new_internal_state[i]
						for sub_partition_new_internal_state in partition_new_internal_states
					]
					for partition_new_internal_states in new_internal_states
				]
				for i in range(batch_count)
			]
		# Properly format for output: action and policy may have multiple heads, swap 1st and 2nd axis
		action_batch = tuple(zip(*action_batch))
		hot_action_batch = tuple(zip(*hot_action_batch))
		policy_batch = tuple(zip(*policy_batch))
		# Return output
		return action_batch, hot_action_batch, policy_batch, value_batch, new_internal_states

	def get_importance_weight(self, info_dict):
		feed_dict = {}
		# Old Policy & Action
		feed_dict.update( self._get_multihead_feed(target=self.old_policy_batch, source=info_dict['policies']) )
		feed_dict.update( self._get_multihead_feed(target=self.old_action_batch, source=info_dict['actions']) )
		if self.has_masked_actions:
			feed_dict.update( self._get_multihead_feed(target=self.old_action_mask_batch, source=info_dict['action_masks']) )
		return tf.get_default_session().run(
			fetches=self.importance_weight_batch, 
			feed_dict=feed_dict
		)
		
	def _get_internal_state(self):
		return tuple(self.network[p].internal_final_state for p in self.get_network_partitions() if self.network[p].use_internal_state)
	
	def _get_internal_state_feed(self, internal_states):
		if not flags.network_has_internal_state:
			return {}
		feed_dict = {}
		i = 0
		for partition in self.get_network_partitions():
			network_partition = self.network[partition]
			if network_partition.use_internal_state:
				partition_batch_states = [
					network_partition.internal_default_state if internal_state is None else internal_state[i]
					for internal_state in internal_states
				]
				for j, initial_state in enumerate(zip(*partition_batch_states)):
					feed_dict.update( {network_partition.internal_initial_state[j]: initial_state} )
				i += 1
		return feed_dict

	def _get_multihead_feed(self, source, target):
		# Action and policy may have multiple heads, swap 1st and 2nd axis of source with zip*
		return { t:s for t,s in zip(target, zip(*source)) }

	def prepare_train(self, info_dict, replay):
		''' Prepare training batch, then _train once using the biggest possible batch '''
		train_type = 1 if replay else 0
		# Get global feed
		current_global_feed = self._big_batch_feed[train_type]
		# Build local feed
		local_feed = self._build_train_feed(info_dict)
		# Merge feed dictionary
		for key,value in local_feed.items():
			if key not in current_global_feed:
				current_global_feed[key] = deque(maxlen=self._train_batch_size) # Initializing the main_feed_dict 
			current_global_feed[key].extend(value)
		# Increase the number of batches composing the big batch
		self._batch_count[train_type] += 1
		if self._batch_count[train_type]%flags.big_batch_size == 0: # can _train
			# Reset batch counter
			self._batch_count[train_type] = 0
			# Reset big-batch (especially if network_has_internal_state) otherwise when in GPU mode it's more time and memory efficient to not reset the big-batch, in order to keep its size fixed
			self._big_batch_feed[train_type] = {}
			# Train
			return self._train(feed_dict=current_global_feed, replay=replay, state_mean_std=(info_dict['state_mean'],info_dict['state_std']))
		return None
	
	def _train(self, feed_dict, replay=False, state_mean_std=None):
		# Add replay boolean to feed dictionary
		feed_dict.update( {self.is_replayed_batch: [replay]} )
		# Intrinsic Reward
		if flags.intrinsic_reward:
			state_mean, state_std = state_mean_std
			feed_dict.update( self._get_multihead_feed(target=self.state_mean_batch, source=[state_mean]) )
			feed_dict.update( self._get_multihead_feed(target=self.state_std_batch, source=[state_std]) )
		# Build _train fetches
		train_tuple = (self.actor_op, self.critic_op) if not replay or flags.train_critic_when_replaying else (self.actor_op, )
		# Do not replay intrinsic reward training otherwise it would start to reward higher the states distant from extrinsic rewards
		if flags.intrinsic_reward and not replay:
			train_tuple += (self.reward_op,)
		if flags.with_transition_predictor:
			train_tuple += (self.transition_predictor_op,)
		# Build fetch
		fetches = [train_tuple] # Minimize loss
		# Get loss values for logging
		fetches += [(self.total_loss, self.policy_loss, self.value_loss)] if flags.print_loss else [()]
		# Debug info
		fetches += [(self.policy_kl_divergence, self.policy_clipping_frequency, self.policy_entropy_regularization)] if flags.print_policy_info else [()]
		# Intrinsic reward
		fetches += [(self.intrinsic_reward_loss, )] if flags.intrinsic_reward else [()]
		# TransitionPredictor
		fetches += [(self.transition_predictor_loss, )] if flags.with_transition_predictor else [()]
		# Run
		_, loss, policy_info, reward_info, transition_predictor_info = tf.get_default_session().run(fetches=fetches, feed_dict=feed_dict)
		self.sync()
		# Build and return loss dict
		train_info = {}
		if flags.print_loss:
			train_info["loss_total"], train_info["loss_actor"], train_info["loss_critic"] = loss
		if flags.print_policy_info:
			train_info["actor_kl_divergence"], train_info["actor_clipping_frequency"], train_info["actor_entropy"] = policy_info
		if flags.intrinsic_reward:
			train_info["intrinsic_reward_loss"] = reward_info
		if flags.with_transition_predictor:
			train_info["transition_predictor_loss"] = transition_predictor_info
		# Build loss statistics
		if train_info:
			self._train_statistics.add(stat_dict=train_info, type='train{}_'.format(self.model_id))
		#=======================================================================
		# if self.loss_distribution_estimator.update([abs(train_info['loss_actor'])]):
		# 	self.actor_loss_is_too_small = self.loss_distribution_estimator.mean <= flags.loss_stationarity_range
		#=======================================================================
		return train_info
		
	def _build_train_feed(self, info_dict):
		# State & Cumulative Return & Old Value
		feed_dict = {
			self.cumulative_return_batch: info_dict['cumulative_returns'],
			self.old_state_value_batch: info_dict['values'],
		}
		if flags.with_transition_predictor:
			feed_dict.update( {self.reward_batch: info_dict['rewards']} )
		feed_dict.update( self._get_multihead_feed(target=self.state_batch, source=info_dict['states']) )
		feed_dict.update( self._get_multihead_feed(target=self.new_state_batch, source=info_dict['new_states']) )
		# Advantage
		feed_dict.update( {self.advantage_batch: info_dict['advantages']} )
		# Old Policy & Action
		feed_dict.update( self._get_multihead_feed(target=self.old_policy_batch, source=info_dict['policies']) )
		feed_dict.update( self._get_multihead_feed(target=self.old_action_batch, source=info_dict['actions']) )
		if self.has_masked_actions:
			feed_dict.update( self._get_multihead_feed(target=self.old_action_mask_batch, source=info_dict['action_masks']) )
		# Internal State
		if flags.network_has_internal_state:
			feed_dict.update( self._get_internal_state_feed([info_dict['internal_state']]) )
			feed_dict.update( {self.size_batch: [len(info_dict['cumulative_returns'])]} )
		return feed_dict

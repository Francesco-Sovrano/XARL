from ray.rllib.utils.framework import try_import_tf
# from ray.rllib.utils.framework import get_activation_fn, try_import_torch
from ray.rllib.models.tf.misc import normc_initializer as tf_normc_initializer
# from ray.rllib.models.torch.misc import normc_initializer as torch_normc_initializer
import gym
import numpy as np
import logging
logger = logging.getLogger(__name__)

tf1, tf, tfv = try_import_tf()
# torch, nn = try_import_torch()

def build_heads(_obs_space, _type, _layers_build_fn, _layers_aggregator_fn):
	_heads = []
	_all_inputs = []
	for _key in filter(lambda x: x.startswith(_type), sorted(_obs_space.original_space.spaces.keys())):
		obs_original_space = _obs_space.original_space[_key]
		if isinstance(obs_original_space, gym.spaces.Dict):
			space_iter = obs_original_space.spaces.items()
			_permutation_invariant = False
		else:
			if not isinstance(obs_original_space, gym.spaces.Tuple):
				obs_original_space = [obs_original_space]
			space_iter = enumerate(obs_original_space)
			_permutation_invariant = True
		_inputs = [
			tf.keras.layers.Input(shape=_head.shape, name=f"{_key}_input_{_name}")
			for _name,_head in space_iter
		]
		_layers = _layers_build_fn(_key,_inputs)
		_heads.append(_layers_aggregator_fn(_key,_layers,_permutation_invariant))
		_all_inputs += _inputs
	return _all_inputs,_heads

def get_tf_heads_model(obs_space):
	cnn_inputs = []
	cnn_heads = []
	fc_inputs = []
	fc_heads = []
	if hasattr(obs_space, 'original_space'):
		def fc_layers_aggregator_fn(_key,_layers,_permutation_invariant): # Permutation invariant aggregator
			assert _layers
			assert _key
			_splitted_units = _key.split('-')
			_units = int(_splitted_units[-1]) if len(_splitted_units) > 1 else 0
			if not _units:
				logger.warning('No units specified: concatenating inputs')
				return tf.keras.layers.Concatenate(axis=-1)(_layers)

			#### FC net
			if len(_layers) <= 1:
				logger.warning(f'Building dense layer with {_units} units on 1 layer')
				return tf.keras.layers.Dense(_units, activation='relu')(_layers[0])

			#### Concat net
			if not _permutation_invariant:
				logger.warning(f'Building concat layer with {_units} units on {len(_layers)} layers')
				_layers = tf.keras.layers.Concatenate(axis=-1)(_layers)
				return tf.keras.layers.Dense(_units, activation='relu')(_layers)

			#### Permutation Invariant net
			logger.warning(f'Building permutation invariant layer with {_units} units on {len(_layers)} layers')
			k = _layers[0].shape[-1]
			_shared_hypernet_layer = tf.keras.Sequential(name=f'shared_hypernet_layer_{_key}', layers=[
				tf.keras.layers.Dense(k*_units, activation='relu'),
				# tf.keras.layers.Dense(k*_units, activation='sigmoid'),
				tf.keras.layers.Reshape((k,_units)),
			])
			_weights = list(map(_shared_hypernet_layer,_layers))
			
			_layers = list(map(tf.keras.layers.Reshape((1, k)), _layers))
			_layers = [
				tf.linalg.matmul(l,w)
				for l,w in zip(_layers,_weights)
			]
			_layers = list(map(tf.keras.layers.Flatten(), _layers))
			return tf.keras.layers.Add()(_layers)
			# _shared_layer = tf.keras.layers.Dense(_units, activation='relu')
			# _layers = list(map(_shared_layer, _layers))
			# return tf.keras.layers.Maximum()(_layers)

		def cnn_layers_build_fn(_key,_inputs):
			return [
				tf.keras.Sequential(name=f"{_key}_layer{i}", layers=[
					tf.keras.layers.Conv2D(name=f'CNN{i}_Conv1',  filters=32, kernel_size=8, strides=4, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_normc_initializer(1.0)),
					tf.keras.layers.Conv2D(name=f'CNN{i}_Conv2',  filters=64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_normc_initializer(1.0)),
					tf.keras.layers.Conv2D(name=f'CNN{i}_Conv3',  filters=64, kernel_size=4, strides=1, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_normc_initializer(1.0)),
					tf.keras.layers.Flatten(),
				])(layer)
				for i,layer in enumerate(_inputs)
			]

		def fc_layers_build_fn(_key,_inputs):
			return [
				tf.keras.layers.Flatten(name=f'flatten{i}_{_key}')(layer)
				for i,layer in enumerate(_inputs)
			]

		cnn_inputs,cnn_heads = build_heads(obs_space, 'cnn', cnn_layers_build_fn, fc_layers_aggregator_fn)
		if len(cnn_heads) > 1: 
			cnn_heads = [tf.keras.layers.Concatenate(axis=-1)(cnn_heads)]
		fc_inputs,fc_heads = build_heads(obs_space, 'fc', fc_layers_build_fn, fc_layers_aggregator_fn)
		if len(fc_heads) > 1: 
			fc_heads = [tf.keras.layers.Concatenate(axis=-1)(fc_heads)]

	last_layer = fc_heads + cnn_heads
	if last_layer:
		if len(last_layer) > 1: last_layer = tf.keras.layers.Concatenate()(last_layer)
		else: last_layer = last_layer[0]
		last_layer = tf.keras.layers.Flatten()(last_layer)
		inputs = cnn_inputs+fc_inputs
	else:
		logger.warning('N.B.: Flattening all observations!')
		last_layer = inputs = tf.keras.layers.Input(shape=(np.prod(obs_space.shape),))
	return inputs, last_layer

def get_heads_input(input_dict):
	obs = input_dict['obs']
	assert isinstance(obs, dict)
	heads_input_list = []
	obs_list = [obs['this']] + obs['others']
	for obs in obs_list:
		cnn_inputs = []
		fc_inputs = []
		other_inputs = []
		for k,v in sorted(obs.items(), key=lambda x:x[0]):
			if k.startswith("cnn"):
				cnn_inputs.append(v)
			elif k.startswith("fc"):
				fc_inputs.append(v)
			else:
				other_inputs.append(v)
		input_list = cnn_inputs + fc_inputs + other_inputs
		flattened_input_list = []
		for i in input_list:
			flattened_input_list += i.values() if isinstance(i,dict) else i
		heads_input_list.append(flattened_input_list)
	return heads_input_list[0], heads_input_list[1:]

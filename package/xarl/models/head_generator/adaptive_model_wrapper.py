from ray.rllib.utils.framework import try_import_tf
# from ray.rllib.utils.framework import get_activation_fn, try_import_torch
from ray.rllib.models.tf.misc import normc_initializer as tf_normc_initializer
# from ray.rllib.models.torch.misc import normc_initializer as torch_normc_initializer
import gym
import numpy as np

tf1, tf, tfv = try_import_tf()
# torch, nn = try_import_torch()

def get_space_iter(obs_original_space):
	if isinstance(obs_original_space, gym.spaces.Dict):
		return obs_original_space.spaces.values()
	if isinstance(obs_original_space, gym.spaces.Tuple):
		return obs_original_space
	return [obs_original_space]

def build_heads(_obs_space, _type, _layers_build_fn, _layers_aggregator_fn):
	_heads = []
	_all_inputs = []
	for _key in filter(lambda x: x.startswith(_type), sorted(_obs_space.original_space.spaces.keys())):
		_inputs = [
			tf.keras.layers.Input(shape=_head.shape, name=f"{_key}_input{i}")
			for i,_head in enumerate(get_space_iter(_obs_space.original_space[_key]))
		]
		_layers = _layers_build_fn(_key,_inputs)
		_heads.append(_layers_aggregator_fn(_key,_layers))
		_all_inputs += _inputs
	return _all_inputs,_heads

def get_tf_heads_model(obs_space):
	cnn_inputs = []
	cnn_heads = []
	fc_inputs = []
	fc_heads = []
	if hasattr(obs_space, 'original_space'):
		# def fc_layers_aggregator_fn(_key,_layers):
		# 	if len(_layers) > 1:
		# 		_layers = []
		# 	_splitted_units = _key.split('_')
		# 	_units = int(_splitted_units[-1]) if len(_splitted_units) > 1 else 0
		# 	return tf.keras.layers.Dense(_units, activation='relu')(_layers[0]) if _units else _layers[0]
		def fc_layers_aggregator_fn(_key,_layers): # Permutation invariant aggregator
			assert _layers
			assert _key
			_splitted_units = _key.split('_')
			_units = int(_splitted_units[-1]) if len(_splitted_units) > 1 else 0
			if not _units:
				return tf.keras.layers.Concatenate(axis=-1)(_layers)

			#### FC net
			if len(_layers) <= 1:
				return tf.keras.layers.Dense(_units, activation='relu')(_layers[0])

			#### Permutation Invariant net
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
		print('N.B.: Flattening all observations!')
		last_layer = inputs = tf.keras.layers.Input(shape=(np.prod(obs_space.shape),))
	return inputs, last_layer

def get_heads_input(input_dict):
	obs = input_dict['obs']
	if not isinstance(obs, dict):
		return input_dict['obs_flat']
	obs_items = sorted(obs.items(), key=lambda x:x[0])
	cnn_inputs = [y for _,y in filter(lambda x: x[0].startswith("cnn"), obs.items())]
	fc_inputs = [y for _,y in filter(lambda x: x[0].startswith("fc"), obs.items())]
	if not isinstance(cnn_inputs,(dict,list)):
		cnn_inputs = [cnn_inputs]
	if not isinstance(fc_inputs,(dict,list)):
		fc_inputs = [fc_inputs]
	if cnn_inputs or fc_inputs:
		if isinstance(cnn_inputs,dict):
			cnn_inputs = list(cnn_inputs.values())
		if isinstance(fc_inputs,dict):
			fc_inputs = list(fc_inputs.values())
		return cnn_inputs + fc_inputs
	return input_dict['obs_flat']

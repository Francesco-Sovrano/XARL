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

def get_tf_heads_model(obs_space):
	cnn_inputs = []
	cnn_layers = []
	fc_inputs = []
	fc_layers = []
	if hasattr(obs_space, 'original_space'):
		if 'cnn' in obs_space.original_space.spaces:
			cnn_inputs = [
				tf.keras.layers.Input(shape=cnn_head.shape, name=f"cnn_input{i}")
				for i,cnn_head in enumerate(get_space_iter(obs_space.original_space['cnn']))
			]
			cnn_layers = [
				tf.keras.Sequential(name=f"cnn_layer{i}", layers=[
					tf.keras.layers.Conv2D(name=f'CNN{i}_Conv1',  filters=32, kernel_size=8, strides=4, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_normc_initializer(1.0)),
					tf.keras.layers.Conv2D(name=f'CNN{i}_Conv2',  filters=64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_normc_initializer(1.0)),
					tf.keras.layers.Conv2D(name=f'CNN{i}_Conv3',  filters=64, kernel_size=4, strides=1, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_normc_initializer(1.0)),
					tf.keras.layers.Flatten(),
				])(layer)
				for i,layer in enumerate(cnn_inputs)
			]
			if len(cnn_layers) > 1:
				cnn_layers = [tf.keras.layers.Concatenate(axis=-1)(cnn_layers)]
		if 'fc' in obs_space.original_space.spaces:
			fc_inputs = [
				tf.keras.layers.Input(shape=fc_head.shape, name=f"fc_input{i}")
				for i,fc_head in enumerate(get_space_iter(obs_space.original_space['fc']))
			]
			fc_layers = [
				tf.keras.layers.Flatten()(layer)
				for layer in fc_inputs
			]
			if len(fc_layers) > 1: 
				fc_layers = [tf.keras.layers.Concatenate(axis=-1)(fc_layers)]

	last_layer = fc_layers + cnn_layers
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
	cnn_inputs = obs.get("cnn",[])
	fc_inputs = obs.get("fc",[])
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

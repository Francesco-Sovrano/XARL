# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
import utils.tensorflow_utils as tf_utils
from agent.network.explicitly_relational_network import ExplicitlyRelational_Network
from utils.interpretable_keras_layers.explicitly_argumentative_layer import ExplicitlyArgumentativeLayer, ExplicitlyRelationalLayer, OR,NOR,AND,NAND,XOR,XNOR
import options
flags = options.get()

class ExplicitlyArgumentative_Network(ExplicitlyRelational_Network):

	def __init__(self, id, policy_heads, scope_dict, training=True, value_count=1, state_scaler=1):
		super().__init__(id, policy_heads, scope_dict, training, value_count, state_scaler)
		self.argument_links = 8
	
	def _relational_layer(self, state, concat, scope="", name="", share_trainables=True):
		layer_type = 'Relational'
		def layer_fn():
			entities = [
				self._entity_extraction_layer(
					features=substate/self.state_scaler, 
					name=f'EE_{i}'+name, 
					share_trainables=share_trainables
				)
				for i,substate in enumerate(state)
			]
			entities = tf.concat(entities,1)
			# Concatenate extra features
			if len(concat) > 0:
				entities = self._concat_layer(
					input=entities, 
					concat=concat,
					share_trainables=share_trainables
				)
			print( "	[{}]Entity Extraction layer {} output shape: {}".format(self.id, name, entities.get_shape()) )
			output = ExplicitlyArgumentativeLayer(
				name='ArgNet1',
				object_pairs=self.object_pairs, 
				edge_size_per_object_pair=self.edge_size_per_object_pair, 
				operators_set=self.relational_layer_operators_set, 
				argument_links=self.argument_links, 
			)(entities)
			relations_set = {}
			return output, relations_set
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)
	
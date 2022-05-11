from ray.rllib.agents.sac.sac_tf_model import SACTFModel

import numpy as np
import gym
from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()


class TFAdaptiveMultiHeadNet:

	@staticmethod
	def init(get_input_layers_and_keras_layers, get_input_list_from_input_dict):
		class TFAdaptiveMultiHeadNetInner(SACTFModel):
			"""
			Data flow:
			`obs` -> forward() (should stay a noop method!) -> `model_out`
			`model_out` -> get_policy_output() -> pi(actions|obs)
			`model_out`, `actions` -> get_q_values() -> Q(s, a)
			`model_out`, `actions` -> get_twin_q_values() -> Q_twin(s, a)
			"""

			def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
				inputs, last_layer = get_input_layers_and_keras_layers(obs_space)
				self.policy_preprocessing_model = tf.keras.Model(inputs, last_layer)
				self.policy_preprocessed_obs_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=last_layer.shape[1:], dtype=np.float32)
				return super().build_policy_model(self.policy_preprocessed_obs_space, num_outputs, policy_model_config, name)

			def build_q_model(self, obs_space, action_space, num_outputs, q_model_config, name):
				inputs, last_layer = get_input_layers_and_keras_layers(obs_space)
				self.value_preprocessing_model = tf.keras.Model(inputs, last_layer)
				self.value_preprocessed_obs_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=last_layer.shape[1:], dtype=np.float32)
				return super().build_q_model(self.value_preprocessed_obs_space, action_space, num_outputs, q_model_config, name)

			def get_policy_output(self, model_out):
				model_out = self.policy_preprocessing_model(get_input_list_from_input_dict({"obs": model_out}))
				return super().get_policy_output(model_out)

			def _get_q_value(self, model_out, actions, net):
				model_out = self.value_preprocessing_model(get_input_list_from_input_dict({"obs": model_out}))
				return super()._get_q_value(model_out, actions, net)

		return TFAdaptiveMultiHeadNetInner

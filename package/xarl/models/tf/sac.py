from ray.rllib.agents.sac.sac_tf_model import SACTFModel

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

			def __init__(self, obs_space, action_space, num_outputs, model_config, name, policy_model_config = None, q_model_config = None, twin_q = False, initial_alpha = 1.0, target_entropy = None):
				inputs, last_layer = get_input_layers_and_keras_layers(obs_space)
				self.preprocessing_model = tf.keras.Model(inputs, last_layer)
				# self.register_variables(self.preprocessing_model.variables)
				self.preprocessed_obs_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=last_layer.shape[1:], dtype=np.float32)
				super().__init__(
					obs_space=obs_space,
					action_space=action_space,
					num_outputs=num_outputs,
					model_config=model_config,
					name=name,
					policy_model_config=policy_model_config,
					q_model_config=q_model_config,
					twin_q=twin_q,
					initial_alpha=initial_alpha,
					target_entropy=target_entropy,
				)

			def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
				return super().build_policy_model(self.preprocessed_obs_space, num_outputs, policy_model_config, name)

			def build_q_model(self, obs_space, action_space, num_outputs, q_model_config, name):
				return super().build_q_model(self.preprocessed_obs_space, action_space, num_outputs, q_model_config, name)

			def _get_q_value(self, model_out, actions, net):
				model_out = self.preprocessing_model(get_input_list_from_input_dict({"obs": model_out}))
				return super()._get_q_value(model_out, actions, net)

			def get_policy_output(self, model_out):
				model_out = self.preprocessing_model(get_input_list_from_input_dict({"obs": model_out}))
				return super().get_policy_output(model_out)

		return TFAdaptiveMultiHeadNetInner

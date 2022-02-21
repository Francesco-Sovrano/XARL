# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import json
import ray
from ray.rllib.models import ModelCatalog
from xarl.utils.workflow import train
from ray.tune.registry import get_trainable_cls, _global_registry, ENV_CREATOR
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from xarl.models.head_generator.primal_adaptive_model_wrapper import get_tf_heads_model, get_heads_input

from environments import *

def get_algorithm_by_name(alg_name):
	# DQN
	if alg_name == 'dqn':
		from ray.rllib.agents.dqn.dqn import DQNTrainer, DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
		from xarl.models.dqn import TFAdaptiveMultiHeadDQN as TFAdaptiveMultiHeadNet
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet.init(get_tf_heads_model, get_heads_input))
		return DQN_DEFAULT_CONFIG.copy(), DQNTrainer
	if alg_name == 'xadqn':
		from xarl.agents.xadqn import XADQNTrainer, XADQN_DEFAULT_CONFIG
		from xarl.models.dqn import TFAdaptiveMultiHeadDQN as TFAdaptiveMultiHeadNet
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet.init(get_tf_heads_model, get_heads_input))
		return XADQN_DEFAULT_CONFIG.copy(), XADQNTrainer
	# DDPG
	if alg_name == 'ddpg':
		from xarl.models.ddpg import TFAdaptiveMultiHeadDDPG as TFAdaptiveMultiHeadNet
		from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_DEFAULT_CONFIG
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet.init(get_tf_heads_model, get_heads_input))
		return DDPG_DEFAULT_CONFIG.copy(), DDPGTrainer
	if alg_name == 'xaddpg':
		from xarl.models.ddpg import TFAdaptiveMultiHeadDDPG as TFAdaptiveMultiHeadNet
		from xarl.agents.xaddpg import XADDPGTrainer, XADDPG_DEFAULT_CONFIG
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet.init(get_tf_heads_model, get_heads_input))
		return XADDPG_DEFAULT_CONFIG.copy(), XADDPGTrainer
	# TD3
	if alg_name == 'td3':
		from xarl.models.ddpg import TFAdaptiveMultiHeadDDPG
		from ray.rllib.agents.ddpg.td3 import TD3Trainer, TD3_DEFAULT_CONFIG
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet.init(get_tf_heads_model, get_heads_input))
		return TD3_DEFAULT_CONFIG.copy(), TD3Trainer
	if alg_name == 'xatd3':
		from xarl.models.ddpg import TFAdaptiveMultiHeadDDPG as TFAdaptiveMultiHeadNet
		from xarl.agents.xaddpg import XATD3Trainer, XATD3_DEFAULT_CONFIG
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet.init(get_tf_heads_model, get_heads_input))
		return XATD3_DEFAULT_CONFIG.copy(), XATD3Trainer
	# SAC
	if alg_name == 'sac':
		from ray.rllib.agents.sac.sac import SACTrainer, DEFAULT_CONFIG as SAC_DEFAULT_CONFIG
		from xarl.models.sac import TFAdaptiveMultiHeadNet
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet.init(get_tf_heads_model, get_heads_input))
		return SAC_DEFAULT_CONFIG.copy(), SACTrainer
	if alg_name == 'xasac':
		from xarl.agents.xasac import XASACTrainer, XASAC_DEFAULT_CONFIG
		from xarl.models.sac import TFAdaptiveMultiHeadNet
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet.init(get_tf_heads_model, get_heads_input))
		return XASAC_DEFAULT_CONFIG.copy(), XASACTrainer
	# PPO
	if alg_name in ['appo','ppo']:
		from ray.rllib.agents.ppo.appo import APPOTrainer, DEFAULT_CONFIG as APPO_DEFAULT_CONFIG
		from xarl.models.appo import TFAdaptiveMultiHeadNet
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet.init(get_tf_heads_model, get_heads_input))
		return APPO_DEFAULT_CONFIG.copy(), APPOTrainer
	if alg_name == 'xappo':
		from xarl.agents.xappo import XAPPOTrainer, XAPPO_DEFAULT_CONFIG
		from xarl.models.appo import TFAdaptiveMultiHeadNet
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet.init(get_tf_heads_model, get_heads_input))
		return XAPPO_DEFAULT_CONFIG.copy(), XAPPOTrainer

import sys
CONFIG, TRAINER = get_algorithm_by_name(sys.argv[1])
ENVIRONMENT = sys.argv[2]
TEST_EVERY_N_STEP = int(float(sys.argv[3]))
STOP_TRAINING_AFTER_N_STEP = int(float(sys.argv[4]))
CENTRALISED_TRAINING = sys.argv[5].lower() == 'true'
NUM_AGENTS = int(float(sys.argv[6]))
if len(sys.argv) > 7:
	print('Updating options..')
	OPTIONS = json.loads(' '.join(sys.argv[7:]))
	print('New options:', CONFIG)
	CONFIG.update(OPTIONS)
CONFIG["callbacks"] = CustomEnvironmentCallbacks

# Setup MARL training strategy: centralised or decentralised
env = _global_registry.get(ENV_CREATOR, ENVIRONMENT)(CONFIG["env_config"])
if isinstance(env,MultiAgentEnv):
	print('Is MultiAgentEnv')
	obs_space = env.observation_space
	act_space = env.action_space
	def gen_policy():
		return (None, obs_space, act_space, {})
	policy_graphs = {}
	if not CENTRALISED_TRAINING:
		for i in range(NUM_AGENTS):
			policy_graphs[f'agent-{i}'] = gen_policy()
		def policy_mapping_fn(agent_id):
				return f'agent-{agent_id}'
	else:
		policy_graphs[f'centralised_agent'] = gen_policy()
		def policy_mapping_fn(agent_id):
				return f'centralised_agent'
	CONFIG.update({
		"multiagent": {
			"policies": policy_graphs,
			"policy_mapping_fn": policy_mapping_fn,
			# Which metric to use as the "batch size" when building a
			# MultiAgentBatch. The two supported values are:
			# env_steps: Count each time the env is "stepped" (no matter how many
			#   multi-agent actions are passed/how many multi-agent observations
			#   have been returned in the previous step).
			# agent_steps: Count each individual agent step as one step.
			"count_steps_by": "agent_steps",
		},
	})
print('Config:', CONFIG)

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True, include_dashboard=False)

train(TRAINER, CONFIG, ENVIRONMENT, test_every_n_step=TEST_EVERY_N_STEP, stop_training_after_n_step=STOP_TRAINING_AFTER_N_STEP)

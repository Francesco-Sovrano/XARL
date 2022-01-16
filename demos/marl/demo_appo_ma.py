# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray
import time
from xarl.utils.workflow import train

from ray.rllib.agents.ppo.appo import APPOTrainer, DEFAULT_CONFIG as APPO_DEFAULT_CONFIG
from environments import *
from xarl.models.appo import TFAdaptiveMultiHeadNet
from ray.rllib.models import ModelCatalog
# Register the models to use.
ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet)

# SELECT_ENV = "Taxi-v3"
# SELECT_ENV = "ToyExample-V0"
# SELECT_ENV = "CescoDrive-V1"
# SELECT_ENV = "GraphDrive-Hard"
# SELECT_ENV = "GridDrive-Hard"
# SELECT_ENV = "Primal"
SELECT_ENV = "Shepherd"

CENTRALISED_TRAINING = True
NUM_AGENTS = 5

CONFIG = APPO_DEFAULT_CONFIG.copy()
CONFIG["env_config"] = {
	'num_dogs': NUM_AGENTS,
	'num_sheep': 50,
}
CONFIG.update({
	"horizon": 2**10, # Number of steps after which the episode is forced to terminate. Defaults to `env.spec.max_episode_steps` (if present) for Gym envs.
	"model": { # this is for GraphDrive and GridDrive
		"custom_model": "adaptive_multihead_network"
	},
	
	"gamma": 0.999, # We use an higher gamma to extend the MDP's horizon; optimal agency on GraphDrive requires a longer horizon.
	"seed": 42, # This makes experiments reproducible.
	"rollout_fragment_length": 2**3, # Number of transitions per batch in the experience buffer. Default is 50 for APPO.
	"train_batch_size": 2**9, # Number of transitions per train-batch. Default is: 100 for TD3, 256 for SAC and DDPG, 32 for DQN, 500 for APPO.
	"replay_proportion": 4, # Set a p>0 to enable experience replay. Saved samples will be replayed with a p:1 proportion to new data samples.
	"replay_buffer_num_slots": 2**14, # Maximum number of batches stored in the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
})
CONFIG["callbacks"] = CustomEnvironmentCallbacks
# framework = CONFIG.get("framework","tf")
# if framework in ["tf2", "tf", "tfe"]:
# 	from ray.rllib.models.tf.fcnet import FullyConnectedNetwork as FCNet, Keras_FullyConnectedNetwork as Keras_FCNet
# elif framework == "torch":
# 	from ray.rllib.models.torch.fcnet import (FullyConnectedNetwork as FCNet)
# ModelCatalog.register_custom_model("fcnet", FCNet)

# Setup MARL training strategy: centralised or decentralised
obs_space = eval(SELECT_ENV)(CONFIG["env_config"]).observation_space
act_space = eval(SELECT_ENV)(CONFIG["env_config"]).action_space
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

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True, include_dashboard=False)

train(XAPPOTrainer, CONFIG, SELECT_ENV, test_every_n_step=1e7, stop_training_after_n_step=4e7)

# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
# os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
import json
import ray
from xarl.utils.workflow import train
from ray.tune.registry import get_trainable_cls, _global_registry, ENV_CREATOR
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from environments import *

from ray.rllib.models import ModelCatalog
from xarl.models import get_model_catalog_dict

def get_algorithm_by_name(alg_name):
	#### DQN
	if alg_name == 'dqn':
		from ray.rllib.agents.dqn.dqn import DQNTrainer, DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
		return DQN_DEFAULT_CONFIG.copy(), DQNTrainer
	if alg_name == 'xadqn':
		from xarl.agents.xadqn import XADQNTrainer, XADQN_DEFAULT_CONFIG
		return XADQN_DEFAULT_CONFIG.copy(), XADQNTrainer
	#### DDPG
	if alg_name == 'ddpg':
		from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_DEFAULT_CONFIG
		return DDPG_DEFAULT_CONFIG.copy(), DDPGTrainer
	if alg_name == 'xaddpg':
		from xarl.agents.xaddpg import XADDPGTrainer, XADDPG_DEFAULT_CONFIG
		return XADDPG_DEFAULT_CONFIG.copy(), XADDPGTrainer
	#### TD3
	if alg_name == 'td3':
		from ray.rllib.agents.ddpg.td3 import TD3Trainer, TD3_DEFAULT_CONFIG
		return TD3_DEFAULT_CONFIG.copy(), TD3Trainer
	if alg_name == 'xatd3':
		from xarl.agents.xaddpg import XATD3Trainer, XATD3_DEFAULT_CONFIG
		return XATD3_DEFAULT_CONFIG.copy(), XATD3Trainer
	#### SAC
	if alg_name == 'sac':
		from ray.rllib.agents.sac.sac import SACTrainer, DEFAULT_CONFIG as SAC_DEFAULT_CONFIG
		return SAC_DEFAULT_CONFIG.copy(), SACTrainer
	if alg_name == 'xasac':
		from xarl.agents.xasac import XASACTrainer, XASAC_DEFAULT_CONFIG
		return XASAC_DEFAULT_CONFIG.copy(), XASACTrainer
	#### PPO
	if alg_name in ['appo','ppo']:
		from ray.rllib.agents.ppo.appo import APPOTrainer, DEFAULT_CONFIG as APPO_DEFAULT_CONFIG
		return APPO_DEFAULT_CONFIG.copy(), APPOTrainer
	if alg_name == 'xappo':
		from xarl.agents.xappo import XAPPOTrainer, XAPPO_DEFAULT_CONFIG
		return XAPPO_DEFAULT_CONFIG.copy(), XAPPOTrainer

import sys
ALG_NAME = sys.argv[1]
CONFIG, TRAINER = get_algorithm_by_name(ALG_NAME)
ENVIRONMENT = sys.argv[2]
EXPERIMENT = None if sys.argv[3].lower()=='none' else sys.argv[3]
TEST_EVERY_N_STEP = int(float(sys.argv[4]))
STOP_TRAINING_AFTER_N_STEP = int(float(sys.argv[5]))
CENTRALISED_TRAINING = sys.argv[6].lower() == 'true'
NUM_AGENTS = int(float(sys.argv[7]))
if len(sys.argv) > 8:
	print('Updating options..')
	OPTIONS = json.loads(' '.join(sys.argv[8:]))
	print('Old options:', CONFIG)
	print('New options:', json.dumps(OPTIONS, indent=4))
	CONFIG.update(OPTIONS)
CONFIG["callbacks"] = CustomEnvironmentCallbacks

for k,v in get_model_catalog_dict(ALG_NAME, CONFIG["framework"]).items():
	ModelCatalog.register_custom_model(k, v)

# Setup MARL training strategy: centralised or decentralised
env = _global_registry.get(ENV_CREATOR, ENVIRONMENT)(CONFIG["env_config"])
obs_space = env.observation_space
act_space = env.action_space
if not CENTRALISED_TRAINING:
	policy_graphs = {
		f'agent-{i}': (None, obs_space, act_space, CONFIG) 
		for i in range(NUM_AGENTS)
	}
	policy_mapping_fn = lambda agent_id: f'agent-{agent_id}'
else:
	policy_graphs = {DEFAULT_POLICY_ID: (None, obs_space, act_space, CONFIG)}
	# policy_graphs = {}
	policy_mapping_fn = lambda agent_id: DEFAULT_POLICY_ID

CONFIG["centralised_buffer"] = CENTRALISED_TRAINING
CONFIG["multiagent"].update({
	"policies": policy_graphs,
	"policy_mapping_fn": policy_mapping_fn,
	# Optional function that can be used to enhance the local agent
	# observations to include more state.
	# See rllib/evaluation/observation_function.py for more info.
	"observation_fn": None,
})
print('Config:', CONFIG)

####################################################################################
####################################################################################

ray.shutdown()
ray.init(
	ignore_reinit_error=True, 
	include_dashboard=False, 
	log_to_driver=False, 
	num_cpus=os.cpu_count(),
)

train(TRAINER, CONFIG, ENVIRONMENT, experiment=EXPERIMENT, test_every_n_step=TEST_EVERY_N_STEP, stop_training_after_n_step=STOP_TRAINING_AFTER_N_STEP)

# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray
import time
from xarl.utils.workflow import train

from ray.rllib.agents.sac.sac import SACTrainer, DEFAULT_CONFIG as SAC_DEFAULT_CONFIG
from environments import *
from ray.rllib.models import ModelCatalog
from xarl.models.sac import TFAdaptiveMultiHeadNet
ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet)

# SELECT_ENV = "CescoDrive-V1"
SELECT_ENV = "GraphDrive-Hard"

CONFIG = SAC_DEFAULT_CONFIG.copy()
CONFIG.update({
	"model": { # this is for GraphDrive and GridDrive
		"custom_model": "adaptive_multihead_network",
	},
	# "preprocessor_pref": "rllib", # this prevents reward clipping on Atari and other weird issues when running from checkpoints
	"gamma": 0.999, # We use an higher gamma to extend the MDP's horizon; optimal agency on GraphDrive requires a longer horizon.
	# "framework": "torch",
	"seed": 42, # This makes experiments reproducible.
	###########################
	"rollout_fragment_length": 1, # Divide episodes into fragments of this many steps each during rollouts. Default is 1.
	"train_batch_size": 2**8, # Number of transitions per train-batch. Default is: 100 for TD3, 256 for SAC and DDPG, 32 for DQN, 500 for APPO.
	# "batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	###########################
	"prioritized_replay": True, # Whether to replay batches with the highest priority/importance/relevance for the agent.
	'buffer_size': 2**14, # Size of the experience buffer. Default 50000
	"prioritized_replay_alpha": 0.6,
	"prioritized_replay_beta": 0.4, # The smaller this is, the stronger is over-sampling
	"prioritized_replay_eps": 1e-6,
	"learning_starts": 2**14, # How many steps of the model to sample before learning starts.
	###########################
	# "grad_clip": 40, # This prevents giant gradients and so improves robustness
	# "l2_reg": 1e-6, # This mitigates over-fitting
	# "normalize_actions": True,
	###########################
	"tau": 1e-4, # The smaller tau, the lower the value over-estimation, the higher the bias
})
CONFIG["callbacks"] = CustomEnvironmentCallbacks

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True, include_dashboard=False)

train(SACTrainer, CONFIG, SELECT_ENV, test_every_n_step=1e7, stop_training_after_n_step=4e7)

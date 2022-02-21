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
from ray.rllib.models import ModelCatalog
from xarl.models.appo import TFAdaptiveMultiHeadNet
from xarl.models.head_generator.adaptive_model_wrapper import get_tf_heads_model, get_heads_input
# Register the models to use.
ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet.init(get_tf_heads_model, get_heads_input))

# SELECT_ENV = "Taxi-v3"
# SELECT_ENV = "ToyExample-V0"
# SELECT_ENV = "CescoDrive-V1"
# SELECT_ENV = "GridDrive-Hard"
SELECT_ENV = "GraphDrive-Hard"

CONFIG = APPO_DEFAULT_CONFIG.copy()
CONFIG.update({
	"model": { # this is for GraphDrive and GridDrive
		"custom_model": "adaptive_multihead_network"
	},
	# "preprocessor_pref": "rllib", # this prevents reward clipping on Atari and other weird issues when running from checkpoints
	"gamma": 0.999, # We use an higher gamma to extend the MDP's horizon; optimal agency on GraphDrive requires a longer horizon.
	"seed": 42, # This makes experiments reproducible.
	"rollout_fragment_length": 2**3, # Number of transitions per batch in the experience buffer. Default is 50 for APPO.
	"train_batch_size": 2**9, # Number of transitions per train-batch. Default is: 100 for TD3, 256 for SAC and DDPG, 32 for DQN, 500 for APPO.
	"replay_proportion": 4, # Set a p>0 to enable experience replay. Saved samples will be replayed with a p:1 proportion to new data samples.
	"replay_buffer_num_slots": 2**14, # Maximum number of batches stored in the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
})
CONFIG["callbacks"] = CustomEnvironmentCallbacks

framework = CONFIG.get("framework","tf")
if framework in ["tf2", "tf", "tfe"]:
	from ray.rllib.models.tf.fcnet import FullyConnectedNetwork as FCNet, Keras_FullyConnectedNetwork as Keras_FCNet
elif framework == "torch":
	from ray.rllib.models.torch.fcnet import (FullyConnectedNetwork as FCNet)
ModelCatalog.register_custom_model("fcnet", FCNet)

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True, include_dashboard=False)

train(APPOTrainer, CONFIG, SELECT_ENV, test_every_n_step=1e7, stop_training_after_n_step=4e7)

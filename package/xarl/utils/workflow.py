# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
import shutil
import time
import xarl.utils.plot_lib as plt
import zipfile
import sys
from io import StringIO
from contextlib import closing
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind, is_atari
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.logger import Logger, UnifiedLogger
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import tempfile
# import json

def find_last_checkpoint_in_directory(directory):
	checkpoint_list = [
		(int(subdir[0].split('_')[-1]), subdir[0])
		for subdir in os.walk(directory)
		if 'checkpoint' in subdir[0]
		and '/' not in subdir[0].split('_')[-1]
	]
	if not checkpoint_list:
		return 0, None
	checkpoint_n, checkpoint_dir = max(checkpoint_list,key=lambda x: x[0])
	for filename in os.listdir(checkpoint_dir):
		file_path = os.path.join(checkpoint_dir, filename)
		if os.path.isfile(file_path):
			if '.' not in filename:
				return checkpoint_n, file_path
	return 0, None

def get_checkpoint_n_logger_by_experiment_id(trainer_class, environment_class, experiment_id):
	if experiment_id is None:
		return 0, None, None
	# timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
	logdir_prefix = "{}_{}_{}".format(trainer_class.__name__, environment_class, experiment_id)
	logdir = os.path.join(DEFAULT_RESULTS_DIR,logdir_prefix) #tempfile.mkdtemp(prefix=logdir_prefix, dir=DEFAULT_RESULTS_DIR)
	if not os.path.exists(logdir):
		os.makedirs(logdir)
	logger_creator_fn = lambda config: UnifiedLogger(config, logdir, loggers=None)
	checkpoint_n, checkpoint = find_last_checkpoint_in_directory(logdir)
	return checkpoint_n, checkpoint, logger_creator_fn

def test(tester_class, config, environment_class, checkpoint, save_gif=True, delete_screens_after_making_gif=True, compress_gif=True, n_episodes=3):
	"""Tests and renders a previously trained model"""
	# test_config = config.copy()
	# test_config['explore'] = False
	agent = tester_class(config, env=environment_class)
	if checkpoint is None:
		raise ValueError(f"A previously trained checkpoint must be provided for algorithm {alg}")
	print(f'Testing with checkpoint: {checkpoint}')
	agent.restore(checkpoint)

	checkpoint_directory = os.path.dirname(checkpoint)
	env = agent.env_creator(config["env_config"])
	# Atari wrapper
	if is_atari(env) and not config.get("custom_preprocessor") and config.get("preprocessor_pref","deepmind") == "deepmind":
		# Deprecated way of framestacking is used.
		framestack = config.get("framestack") is True
		# framestacking via trajectory view API is enabled.
		num_framestacks = config.get("num_framestacks", 0)

		# Trajectory view API is on and num_framestacks=auto:
		# Only stack traj. view based if old
		# `framestack=[invalid value]`.
		if num_framestacks == "auto":
			if framestack == DEPRECATED_VALUE:
				config["num_framestacks"] = num_framestacks = 4
			else:
				config["num_framestacks"] = num_framestacks = 0
		framestack_traj_view = num_framestacks > 1
		env = wrap_deepmind(
			env,
			# dim=config.get("dim"),
			framestack=framestack,
			framestack_via_traj_view_api=framestack_traj_view
		)

	render_modes = env.metadata['render.modes']
	env.seed(config["seed"]+1)
	def print_screen(screens_directory, step):
		filename = os.path.join(screens_directory, f'frame{step}.jpg')
		if 'rgb_array' in render_modes:
			plt.rgb_array_image(
				env.render(mode='rgb_array'), 
				filename
			)
		elif 'ansi' in render_modes:
			plt.ascii_image(
				env.render(mode='ansi'), 
				filename
			)
		elif 'ascii' in render_modes:
			plt.ascii_image(
				env.render(mode='ascii'), 
				filename
			)
		elif 'human' in render_modes:
			old_stdout = sys.stdout
			sys.stdout = StringIO()
			env.render(mode='human')
			with closing(sys.stdout):
				plt.ascii_image(
					sys.stdout.getvalue(), 
					filename
				)
			sys.stdout = old_stdout
		else:
			raise Exception(f"No compatible render mode (rgb_array,ansi,ascii,human) in {render_modes}.")
		return filename

	multiagent = isinstance(env, MultiAgentEnv)
	for episode_id in range(n_episodes):
		episode_directory = os.path.join(checkpoint_directory, f'episode_{episode_id}')
		os.mkdir(episode_directory)
		screens_directory = os.path.join(episode_directory, 'screen')
		os.mkdir(screens_directory)
		log_list = []
		sum_reward = 0
		step = 0
		
		if multiagent:
			policy_mapping_fn = config["multiagent"]["policy_mapping_fn"]
			state_dict = env.reset()
			done_dict = {i: False for i in state_dict.keys()}
			done_dict['__all__'] = False
			# state_dict = dict(zip(state_dict.keys(),map(np.squeeze,state_dict.values())))
			file_list = [print_screen(screens_directory, step)]
			while not done_dict['__all__'] and (not config.get('horizon',None) or step < config['horizon']):
				step += 1
				# action = env.action_space.sample()
				action_dict = {
					k: agent.compute_action(v, policy_id=policy_mapping_fn(k), full_fetch=True, explore=False)[0]
					for k,v in state_dict.items()
					if not done_dict.get(k, True)
				}
				# print('action_dict', action_dict)
				state_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
				# state_dict = dict(zip(state_dict.keys(),map(np.squeeze,state_dict.values())))
				sum_reward += sum(reward_dict.values())
				file_list.append(print_screen(screens_directory, step))
				log_list.append(', '.join([
					f'step: {step}',
					f'reward: {reward_dict}',
					f'done: {done_dict}',
					f'info: {info_dict}',
					f'action: {action_dict}',
					f'state: {state_dict}',
					'\n\n',
				]))
		else:
			done = False
			state = np.squeeze(env.reset())
			file_list = [print_screen(screens_directory, step)]
			while not done and (not config.get('horizon',None) or step < config['horizon']):
				step += 1
				# action = env.action_space.sample()
				action = agent.compute_action(state, full_fetch=True, explore=False)
				state, reward, done, info = env.step(action[0])
				state = np.squeeze(state)
				sum_reward += reward
				file_list.append(print_screen(screens_directory, step))
				log_list.append(', '.join([
					f'step: {step}',
					f'reward: {reward}',
					f'done: {done}',
					f'info: {info}',
					f'action: {action}',
					f'state: {state}',
					'\n\n',
				]))
		with open(os.path.join(episode_directory, f'episode_{step}_{sum_reward}.log'), 'w') as f:
			f.writelines(log_list)
		if save_gif:
			gif_file_name = f'episode_{step}_{sum_reward}.gif'
			gif_file_path = os.path.join(episode_directory, gif_file_name)
			plt.make_gif(file_list=file_list, gif_path=gif_file_path)
			# Delete screens, to save memory
			if delete_screens_after_making_gif:
				shutil.rmtree(screens_directory, ignore_errors=True)
			# Zip GIF, to save memory
			if compress_gif:
				with zipfile.ZipFile(gif_file_path+'.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as z:
					z.write(gif_file_path,gif_file_name)
				# Remove unzipped GIF
				os.remove(gif_file_path)

def train(trainer_class, config, environment_class, experiment=None, test_every_n_step=None, stop_training_after_n_step=None, log=True):
	_, checkpoint, logger_creator_fn = get_checkpoint_n_logger_by_experiment_id(trainer_class, environment_class, experiment)
	# Add required Multi-Agent XAER options
	if config.get("clustering_scheme", None):
		if 'Who' in config["clustering_scheme"]:
			config["env_config"]['build_action_list'] = True
			print('Added "build_action_list" to "env_config"')
		if 'Which_CoopStrategy' in config["clustering_scheme"]:
			config["env_config"]['build_joint_action_list'] = True
			print('Added "build_joint_action_list" to "env_config"')
	# Configure RLlib to train a policy using the given environment and trainer
	agent = trainer_class(config, env=environment_class, logger_creator=logger_creator_fn)
	if checkpoint:
		print(f'Loading checkpoint: {checkpoint}')
		agent.restore(checkpoint)
	# Inspect the trained policy and model, to see the results of training in detail
	policy = agent.get_policy()
	if not policy:
		policy_list = [
			agent.get_policy(policy_id)
			for policy_id in config["multiagent"]["policies"].keys()
		]
	else:
		policy_list = [policy]
	for i,policy in enumerate(policy_list):
		model = policy.model
		print(f'Members of model of agent with ID {i}:', dir(model))
		if hasattr(model, 'action_space'):
			print('#'*10)
			print(f'action_space of agent with ID {i}:')
			print(model.action_space)
			print('#'*10)
		if hasattr(model, 'obs_space'):
			print('#'*10)
			print(f'obs_space of agent with ID {i}:')
			print(model.obs_space)
			print('#'*10)
	# Start training
	n = 0
	sample_steps = 0
	if stop_training_after_n_step is None:
		stop_training_after_n_step = float('inf')
	check_steps = test_every_n_step if test_every_n_step is not None else float('inf')
	def save_checkpoint():
		checkpoint = agent.save()
		print(f'Checkpoint saved in {checkpoint}')
		print(f'Testing..')
		try:
			test(trainer_class, config, environment_class, checkpoint)
		except Exception as e:
			print(e)
		
	while sample_steps < stop_training_after_n_step:
		n += 1
		last_time = time.time()
		result = agent.train()
		train_steps = result["info"]["num_steps_trained"]
		sample_steps = result["info"]["num_steps_sampled"]
		episode = {
			'n': n, 
			'episode_reward_min': result['episode_reward_min'], 
			'episode_reward_mean': result['episode_reward_mean'], 
			'episode_reward_max': result['episode_reward_max'],  
			'episode_len_mean': result['episode_len_mean']
		}
		if log:
			print(', '.join([
				f'iteration: {n+1}',
				f'episode_reward (min/mean/max): {result["episode_reward_min"]:.2f}/{result["episode_reward_mean"]:.2f}/{result["episode_reward_max"]:.2f}',
				f'episode_len_mean: {result["episode_len_mean"]:.2f}',
				f'steps_trained: {train_steps}',
				f'steps_sampled: {sample_steps}',
				f'train_ratio: {(train_steps/sample_steps):.2f}',
				f'seconds: {time.time()-last_time:.2f}'
			]))
		if sample_steps>=stop_training_after_n_step:
			save_checkpoint()
		elif n == 1:
			check_steps = sample_steps + test_every_n_step
		elif sample_steps>=check_steps:
			check_steps += test_every_n_step
			save_checkpoint()

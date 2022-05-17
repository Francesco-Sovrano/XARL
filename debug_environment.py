import gym
import time
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import xarl.utils.plot_lib as plt
import sys
import os
from environments import *

HORIZON = 2**7
PLOT_EPISODE = False
if PLOT_EPISODE:
	OUTPUT_DIR = './demo_episode'
	os.makedirs(OUTPUT_DIR, exist_ok=True)

env_config = {
	'num_agents': 64,
	'force_car_to_stay_on_road': True,
	'optimal_steering_angle_on_road': True,
	'visibility_radius': 10,
	'max_food_per_target': 1,
	'blockage_probability': None,
	# 'blockage_probability': 0.15,
	# 'min_blockage_ratio': 0.1,
	# 'max_blockage_ratio': 0.5,
	'agent_collision_radius': None,
	'target_junctions_number': 9,
	'source_junctions_number': 2,
	################################
	'max_dimension': 64,
	'junctions_number': 64,
	'max_roads_per_junction': 4,
	'junction_radius': 1,
	'max_distance_to_path': .5, # meters
	################################
	'random_seconds_per_step': False, # whether to sample seconds_per_step from an exponential distribution
	'mean_seconds_per_step': 0.5, # in average, a step every n seconds
	################################
	# track = 0.4 # meters # https://en.wikipedia.org/wiki/Axle_track
	'wheelbase': 0.15, # meters # https://en.wikipedia.org/wiki/Wheelbase
	# information about speed parameters: http://www.ijtte.com/uploads/2012-10-01/5ebd8343-9b9c-b1d4IJTTE%20vol2%20no3%20%287%29.pdf
	'min_speed': 0.2, # m/s
	'max_speed': 1.2, # m/s
	# the fastest car has max_acceleration 9.25 m/s^2 (https://en.wikipedia.org/wiki/List_of_fastest_production_cars_by_acceleration)
	# the slowest car has max_acceleration 0.7 m/s^2 (http://automdb.com/max_acceleration)
	'max_acceleration': 1, # m/s^2
	# the best car has max_deceleration 29.43 m/s^2 (https://www.quora.com/What-can-be-the-maximum-deceleration-during-braking-a-car?share=1)
	# a normal car has max_deceleration 7.1 m/s^2 (http://www.batesville.k12.in.us/Physics/PhyNet/Mechanics/Kinematics/BrakingDistData.html)
	'max_deceleration': 7, # m/s^2
	'max_steering_degree': 45,
	# min_speed_lower_limit = 0.7 # m/s # used together with max_speed to get the random speed upper limit
	# max_speed_noise = 0.25 # m/s
	# max_steering_noise_degree = 2
	'max_speed_noise': 0, # m/s
	'max_steering_noise_degree': 0,
	'max_normalised_speed': 120,
}

env = PVCommMultiAgentGraphDrive({"reward_fn": 'frequent_reward_no_culture', "culture_level": None, **env_config})
env.seed(38)
# env = CescoDriveV0()
multiagent = isinstance(env, MultiAgentEnv)
render_modes = env.metadata['render.modes']

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

def run_one_episode(env):
	state = env.reset()
	step = 0
	sum_reward = 0
	if PLOT_EPISODE:
		file_list = [print_screen(OUTPUT_DIR, step)]
	if multiagent:
		done_dict = {i: False for i in state.keys()}
		done_dict['__all__'] = False
		while not done_dict['__all__'] and step <= HORIZON:
			t = time.time()
			step += 1
			action_dict = {
				i: env.action_space.sample()
				for i in state.keys()
				if not done_dict.get(i,True)
			}
			state_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
			state = state_dict
			sum_reward += sum(reward_dict.values())
			print(f'step {step} took {time.time()-t:.2f} seconds')
			if PLOT_EPISODE:
				file_list.append(print_screen(OUTPUT_DIR, step))
			env.render()
			# time.sleep(0.25)
	else:
		done = False
		while not done and step <= HORIZON:
			t = time.time()
			step += 1
			action = env.action_space.sample()
			state, reward, done, info = env.step(action)
			sum_reward += reward
			print(f'step {step} took {time.time()-t:.2f} seconds')
			if PLOT_EPISODE:
				file_list.append(print_screen(OUTPUT_DIR, step))
			env.render()
			# time.sleep(0.25)
	if PLOT_EPISODE:
		gif_filename = os.path.join(OUTPUT_DIR, 'episode.gif')
		plt.make_gif(file_list=file_list, gif_path=gif_filename)
	return sum_reward

sum_reward = run_one_episode(env)
sum_reward = run_one_episode(env)

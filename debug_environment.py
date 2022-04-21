import gym
import time
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from environments import *

env_config = {
	'num_agents': 5,
	'max_visit_per_junction': 2,
	'mean_blockage': 0.1,
	'agent_collision_radius': None,
	'random_seconds_per_step': False, # whether to sample seconds_per_step from an exponential distribution
	'mean_seconds_per_step': 0.25, # in average, a step every n seconds
	# track = 0.4 # meters # https://en.wikipedia.org/wiki/Axle_track
	'wheelbase': 0.35, # meters # https://en.wikipedia.org/wiki/Wheelbase
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
	# max_step = 2**9
	'max_distance_to_path': 0.5, # meters
	# min_speed_lower_limit = 0.7 # m/s # used together with max_speed to get the random speed upper limit
	# max_speed_noise = 0.25 # m/s
	# max_steering_noise_degree = 2
	'max_speed_noise': 0, # m/s
	'max_steering_noise_degree': 0,
	# multi-road related stuff
	'max_dimension': 50,
	'junction_number': 32,
	'max_roads_per_junction': 4,
	'junction_radius': 1,
	'max_normalised_speed': 120,
}
env = MultiAgentGraphDrive({"reward_fn": 'frequent_reward_default', "culture_level": "Hard", **env_config})
# env = CescoDriveV0()
multiagent = isinstance(env, MultiAgentEnv)

def run_one_episode (env):
	env.seed(38)
	state = env.reset()
	sum_reward = 0
	done = False
	if multiagent:
		agent_id_list = list(state.keys())
		while not done:
			action_dict = {
				i: env.action_space.sample()
				for i in agent_id_list
			}
			state_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
			sum_reward += sum(reward_dict.values())
			done = done_dict['__all__']
			env.render()
			time.sleep(0.5)
	else:
		while not done:
			action = env.action_space.sample()
			state, reward, done, info = env.step(action)
			sum_reward += reward
			env.render()
			time.sleep(0.5)
	return sum_reward

sum_reward = run_one_episode(env)
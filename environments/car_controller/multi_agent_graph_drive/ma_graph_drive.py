# -*- coding: utf-8 -*-
import gym
from gym.utils import seeding
import numpy as np
import json
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from matplotlib import use as matplotlib_use
matplotlib_use('Agg',force=True) # no display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

from environments.car_controller.utils.geometry import *
from environments.car_controller.graph_drive.lib.roads import RoadNetwork
from environments.car_controller.multi_agent_graph_drive.ma_roads import MultiAgentRoadNetwork
from environments.car_controller.grid_drive.lib.road_cultures import *

import logging
logger = logging.getLogger(__name__)

def get_normalized_visit_count(j, max_visit_per_junction):
	return np.clip(j.visit_count, 0, max_visit_per_junction)/max_visit_per_junction

class GraphDriveAgent:

	def seed(self, seed=None):
		logger.warning(f"Setting random seed to: {seed}")
		self.np_random, _ = seeding.np_random(seed)
		return [seed]

	def __init__(self, n_of_other_agents, culture, env_config):
		super().__init__()
		
		self.n_of_other_agents = n_of_other_agents
		self.env_config = env_config
		self.reward_fn = eval(f'self.{self.env_config["reward_fn"]}')
		
		self.culture = culture
		self.obs_road_features = len(self.culture.properties)  # Number of binary ROAD features in Hard Culture
		self.obs_car_features = len(self.culture.agent_properties) - 1  # Number of binary CAR features in Hard Culture (excluded speed)
		# Spaces
		self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # steering angle and speed
		state_dict = {
			"junctions_view": gym.spaces.Box( # Closest road to the agent (the one it's driving on), sorted by relative position
				low= -1,
				high= 1,
				shape= (
					self.env_config['junction_number'],
					2 + 1, # junction.pos + junction.normalized_visit_count
				),
				dtype=np.float32
			),
			"roads_view": gym.spaces.Box( # Roads directly connected to the closest road to the agent (the one it's driving on), sorted by relative position
				low= -1,
				high= 1,
				shape= ( # closest junctions view
					self.env_config['junction_number'], # number of junctions
					self.env_config['max_roads_per_junction'], # maximum number of roads per junction
					2 + 2 + self.obs_road_features,  # road properties: road.start.pos + road.end.pos + road.af_features
				),
				dtype=np.float32
			),
			"agent_features": gym.spaces.Box( # Agent features
				low= -1,
				high= 1,
				shape= (
					self.agent_state_size + self.obs_car_features,
				),
				dtype=np.float32
			),
		}
		if self.n_of_other_agents > 0:
			state_dict["agents_view"] = gym.spaces.Box( # Agent features
				low= -1,
				high= 1,
				shape= (
					self.n_of_other_agents, 
					2 + self.agent_state_size + self.obs_car_features + 1
				), # for each other possible agent give position + heading vector + features with no access to state + is_dead
				dtype=np.float32
			)
		self.observation_space = gym.spaces.Dict({
			"fc": gym.spaces.Dict(state_dict),
		})

	def reset(self, car_point, agent_id, road_network, other_agent_list):
		self.agent_id = agent_id
		self.road_network = road_network
		self.other_agent_list = other_agent_list
		self.seconds_per_step = self.get_step_seconds()
		self.slowed_down = self.is_blocked()
		# car position
		self.car_point = car_point
		self.car_orientation = (2*self.np_random.random()-1)*np.pi # in [-pi,pi]
		self.distance_to_closest_road, self.closest_road, self.closest_junction_list = self.road_network.get_closest_road_and_junctions(self.car_point)
		self.closest_junction = RoadNetwork.get_closest_junction(self.closest_junction_list, self.car_point)
		# steering angle & speed
		self.steering_angle = 0
		self.speed = self.env_config['min_speed'] #+ (self.env_config['max_speed']-self.env_config['min_speed'])*self.np_random.random() # in [min_speed,max_speed]
		# self.speed = self.env_config['min_speed']+(self.env_config['max_speed']-self.env_config['min_speed'])*(70/120) # for testing
		self.agent_id.assign_property_value("Speed", self.road_network.normalise_speed(self.env_config['min_speed'], self.env_config['max_speed'], self.speed))

		self.last_closest_road = None
		self.last_closest_junction = None
		self.goal_junction = None
		self.current_road_speed_list = []
		# init concat variables
		self.last_reward = 0
		self.last_reward_type = 'move_forward'
		self.last_action_mask = None
		self.is_dead = False

	@property
	def normalised_speed(self):
		# return (self.speed-self.env_config['min_speed']*0.9)/(self.env_config['max_speed']-self.env_config['min_speed']*0.9) # in (0,1]
		return self.speed/self.env_config['max_speed'] # in (0,1]

	def get_state(self, car_point=None, car_orientation=None):
		if car_point is None:
			car_point=self.car_point
		if car_orientation is None:
			car_orientation=self.car_orientation
		junctions_view, roads_view, agents_view = self.get_view(car_point, car_orientation)
		state_dict = {
			"junctions_view": junctions_view,
			"roads_view": roads_view,
			"agent_features": np.array([
				*self.get_agent_state(),
				*self.agent_id.binary_features(as_tuple=True), 
			], dtype=np.float32)
		}
		if self.n_of_other_agents > 0:
			state_dict["agents_view"] = agents_view
		return {
			"fc": state_dict
		}

	@property
	def agent_state_size(self):
		return 5 # normalised steering angle + normalised speed	

	def get_agent_state(self):
		return (
			self.steering_angle/self.env_config['max_steering_angle'], # normalised steering angle
			self.speed/self.env_config['max_speed'], # normalised speed
			min(1, self.distance_to_closest_road/self.env_config['max_distance_to_path']),
			self.is_in_junction(self.car_point),
			1 if self.slowed_down else 0
		)

	def normalize_point(self, p):
		return (np.clip(p[0]/self.env_config['map_size'][0],-1,1), np.clip(p[1]/self.env_config['map_size'][1],-1,1))

	def colliding_with_other_agent(self, old_car_point, car_point):
		if not self.env_config['agent_collision_radius']:
			return False
		for agent in self.other_agent_list:
			if segment_collide_circle(segment=(old_car_point, car_point), circle=(agent.car_point,self.env_config['agent_collision_radius'])):
				return True
		return False

	def get_normalized_visit_count(self, j):
		return get_normalized_visit_count(j, self.env_config['max_visit_per_junction'])

	def get_view(self, source_point, source_orientation): # source_orientation is in radians, source_point is in meters, source_position is quantity of past splines
		source_x, source_y = source_point
		shift_rotate_normalise_point = lambda x: self.normalize_point(shift_and_rotate(*x, -source_x, -source_y, -source_orientation))
		road_network_junctions = filter(lambda j: j.roads_connected, self.road_network.junctions)
		road_network_junctions = map(lambda j: (shift_rotate_normalise_point(j.pos),j), road_network_junctions)
		sorted_junctions = sorted(road_network_junctions, key=lambda x: x[0])
		# Get junctions view
		junctions_view = np.full(self.observation_space['fc']["junctions_view"].shape, -1, dtype=np.float32)
		junctions_view[:len(sorted_junctions)] = [
			(*j_pos, self.get_normalized_visit_count(j))
			for j_pos, j in sorted_junctions
		]
		# Get roads view
		roads_view = np.full(self.observation_space['fc']["roads_view"].shape, -1, dtype=np.float32)
		for i,(_,j) in enumerate(sorted_junctions):
			roads_view[i,:len(j.roads_connected)] = sorted(
				(
					(
						*shift_rotate_normalise_point(road.start.pos),
						*shift_rotate_normalise_point(road.end.pos),
						*road.binary_features(as_tuple=True), # in [0,1]
						# 1 if road.is_visited_by(self.agent_id) else 0, # whether road has been previously visited
					) if euclidean_distance(road.start.pos,j.pos) < euclidean_distance(road.end.pos,j.pos) else (
						*shift_rotate_normalise_point(road.end.pos),
						*shift_rotate_normalise_point(road.start.pos),
						*road.binary_features(as_tuple=True), # in [0,1]
						# 1 if road.is_visited_by(self.agent_id) else 0, # whether road has been previously visited
					)
					for road in j.roads_connected
				), 
				key=lambda x:(x[0:4])
			)
		##### Get neighbourhood view
		if self.other_agent_list:
			sorted_agents = sorted((
				(shift_rotate_normalise_point(agent.car_point), agent.get_agent_state(), agent.agent_id.binary_features(as_tuple=True), agent.is_dead)
				for agent in self.other_agent_list
			), key=lambda x: x[0])
			agents_view = np.array(
				[
					(*agent_point, *agent_state, *agent_features, 1 if agent_is_dead else 0)
					for agent_point, agent_state, agent_features, agent_is_dead in sorted_agents
				]
			, dtype=np.float32)
		else:
			agents_view = None
		return junctions_view, roads_view, agents_view

	def move(self, point, orientation, steering_angle, speed, add_noise=False):
		# https://towardsdatascience.com/how-self-driving-cars-steer-c8e4b5b55d7f?gi=90391432aad7
		# Add noise
		if add_noise:
			steering_angle += (2*self.np_random.random()-1)*self.env_config['max_steering_noise_angle']
			steering_angle = np.clip(steering_angle, -self.env_config['max_steering_angle'], self.env_config['max_steering_angle']) # |steering_angle| <= max_steering_angle, ALWAYS
			speed += (2*self.np_random.random()-1)*self.env_config['max_speed_noise']
		#### Ackerman Steering: Forward Kinematic for Car-Like vehicles #### https://www.xarg.org/book/kinematics/ackerman-steering/
		turning_radius = self.env_config['wheelbase']/np.tan(steering_angle)
		# Max taylor approximation error of the tangent simplification is about 3° at 30° steering lock
		# turning_radius = self.env_config['wheelbase']/steering_angle
		angular_velocity = speed/turning_radius
		# get normalized new orientation
		new_orientation = np.mod(orientation + angular_velocity*self.seconds_per_step, 2*np.pi) # in [0,2*pi)
		# Move point
		x, y = point
		dir_x, dir_y = get_heading_vector(angle=new_orientation, space=speed*self.seconds_per_step)
		return (x+dir_x, y+dir_y), new_orientation

	def get_steering_angle_from_action(self, action): # action is in [-1,1]
		return action*self.env_config['max_steering_angle'] # in [-max_steering_angle, max_steering_angle]
		
	def get_acceleration_from_action(self, action): # action is in [-1,1]
		return action*(self.env_config['max_acceleration'] if action >= 0 else self.env_config['max_deceleration']) # in [-max_deceleration, max_acceleration]
		
	def accelerate(self, speed, acceleration):
		# use seconds_per_step instead of mean_seconds_per_step, because this way the algorithm is able to explore more states and train better
		# return np.clip(speed + acceleration*self.env_config['mean_seconds_per_step'], self.env_config['min_speed'], self.env_config['max_speed'])
		return np.clip(speed + acceleration*self.seconds_per_step, self.env_config['min_speed'], self.env_config['max_speed'])
		
	def is_in_junction(self, car_point, radius=None):
		if radius is None:
			radius = self.env_config['junction_radius']
		return euclidean_distance(self.closest_junction.pos, car_point) <= radius

	def get_step_seconds(self):
		return self.np_random.exponential(scale=self.env_config['mean_seconds_per_step']) if self.env_config['random_seconds_per_step'] is True else self.env_config['mean_seconds_per_step']

	def is_blocked(self):
		if not self.env_config['mean_blockage']:
			return False
		return self.np_random.choice(a=[False,True], p=[1-self.env_config['mean_blockage'],self.env_config['mean_blockage']])

	def start_step(self, action_vector):
		# first of all, get the seconds passed from last step
		self.seconds_per_step = self.get_step_seconds()
		# compute new steering angle
		self.steering_angle = self.get_steering_angle_from_action(action=action_vector[0])
		# compute new acceleration
		self.acceleration = self.get_acceleration_from_action(action=action_vector[1])
		# compute new speed
		self.speed = self.accelerate(speed=self.speed, acceleration=self.acceleration)
		self.agent_id.assign_property_value("Speed", self.road_network.normalise_speed(self.env_config['min_speed'], self.env_config['max_speed'], self.speed))
		# move car
		old_car_point = self.car_point
		old_goal_junction = self.goal_junction
		visiting_new_road = False
		visiting_new_junction = False
		self.car_point, self.car_orientation = self.move(
			point=self.car_point, 
			orientation=self.car_orientation, 
			steering_angle=self.steering_angle, 
			speed=self.speed/2 if self.slowed_down else self.speed, 
			add_noise=True
		)
		if self.goal_junction is None:
			self.distance_to_closest_road, self.closest_road, self.closest_junction_list = self.road_network.get_closest_road_and_junctions(self.car_point, self.closest_junction_list)
		else:
			self.distance_to_closest_road = point_to_line_dist(self.car_point, self.closest_road.edge)
		self.closest_junction = RoadNetwork.get_closest_junction(self.closest_junction_list, self.car_point)
		# if a new road is visited, add the old one to the set of visited ones
		if self.is_in_junction(self.car_point):
			if self.closest_junction != self.last_closest_junction:
				visiting_new_junction = True
				self.closest_junction.visit()
				self.last_closest_junction = self.closest_junction
			self.goal_junction = None
			if self.last_closest_road is not None: # if closest_road is not the first visited road
				self.last_closest_road.is_visited_by(self.agent_id, True) # set the old road as visited
		elif self.last_closest_road != self.closest_road: # not in junction and visiting a new road
			visiting_new_road = True
			self.last_closest_road = self.closest_road # keep track of the current road
			self.goal_junction = RoadNetwork.get_furthest_junction(self.closest_junction_list, self.car_point)
			self.current_road_speed_list = []
		self.current_road_speed_list.append(self.speed)
		return visiting_new_road, visiting_new_junction, old_goal_junction, old_car_point

	def end_step(self, visiting_new_road, visiting_new_junction, old_goal_junction, old_car_point):
		# compute perceived reward
		reward, dead, reward_type = self.reward_fn(visiting_new_road, visiting_new_junction, old_goal_junction, old_car_point)
		self.slowed_down = self.is_blocked()
		# compute new state (after updating progress)
		state = self.get_state()
		# update last action/state/reward
		self.last_reward = reward
		self.last_reward_type = reward_type
		info_dict = {'explanation':{'why':reward_type}}
		self.is_dead = dead
		return [state, reward, dead, info_dict]
			
	def get_info(self):
		return f"speed={self.speed}, steering_angle={self.steering_angle}, orientation={self.car_orientation}"

	def frequent_reward_default(self, visiting_new_road, visiting_new_junction, old_goal_junction, old_car_point): # BAD
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			reward = self.normalised_speed # in (0,1]
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		#######################################
		# "Is colliding" rule
		if self.colliding_with_other_agent(old_car_point, self.car_point):
			return unitary_reward(is_positive=False, is_terminal=True, label='is_colliding_other_agent')

		is_in_junction = self.is_in_junction(self.car_point)
		#######################################
		# "Is in junction" rule
		if is_in_junction:
			#######################################
			# "Is in new junction" rule
			if visiting_new_junction and self.get_normalized_visit_count(self.closest_junction) < 1:  # If agent acquired a brand new junction and that is important.
				# return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('is_in_new_junction', self.last_explanation_list))
				return unitary_reward(is_positive=True, is_terminal=False, label='has_visited_an_important_junction')
			#######################################
			# "Is in old junction" rule
			return null_reward(is_terminal=False, label='is_in_junction')
		#######################################
		# "No U-Turning outside junction" rule
		space_traveled_towards_goal = euclidean_distance(self.goal_junction.pos, old_car_point) - euclidean_distance(self.goal_junction.pos, self.car_point) if self.goal_junction is not None else 0
		if space_traveled_towards_goal <= 0:
			return unitary_reward(is_positive=False, is_terminal=True, label='u_turning_outside_junction')
		#######################################
		# "Stay on the road" rule
		if self.distance_to_closest_road >= self.env_config['max_distance_to_path']:
			return unitary_reward(is_positive=False, is_terminal=True, label='not_staying_on_the_road')
		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		# Assign normalised speed to agent properties before running dialogues.
		following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.agent_id, explanation_type="compact")
		if not following_regulation:
			return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation', explanation_list))
		#######################################
		# "Move towards important junction" rule
		if self.get_normalized_visit_count(self.goal_junction) < 1:
			return step_reward(is_positive=True, is_terminal=False, label='moving_towards_important_junction')
		#######################################
		# "Move forward" rule
		return null_reward(is_terminal=False, label='moving_forward')

	def sparse_reward_default(self, visiting_new_road, visiting_new_junction, old_goal_junction, old_car_point): # BAD
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		#######################################
		# "Is colliding" rule
		if self.colliding_with_other_agent(old_car_point, self.car_point):
			return unitary_reward(is_positive=False, is_terminal=True, label='is_colliding_other_agent')

		is_in_junction = self.is_in_junction(self.car_point)
		#######################################
		# "Is in junction" rule
		if is_in_junction:
			#######################################
			# "Is in new junction" rule
			if visiting_new_junction and self.get_normalized_visit_count(self.closest_junction) < 1:  # If agent acquired a brand new junction and that is important.
				# return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('is_in_new_junction', self.last_explanation_list))
				return unitary_reward(is_positive=True, is_terminal=False, label='has_visited_an_important_junction')
			#######################################
			# "Is in old junction" rule
			return null_reward(is_terminal=False, label='is_in_junction')
		#######################################
		# "No U-Turning outside junction" rule
		space_traveled_towards_goal = euclidean_distance(self.goal_junction.pos, old_car_point) - euclidean_distance(self.goal_junction.pos, self.car_point) if self.goal_junction is not None else 0
		if space_traveled_towards_goal <= 0:
			return unitary_reward(is_positive=False, is_terminal=True, label='u_turning_outside_junction')
		#######################################
		# "Stay on the road" rule
		if self.distance_to_closest_road >= self.env_config['max_distance_to_path']:
			return unitary_reward(is_positive=False, is_terminal=True, label='not_staying_on_the_road')
		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		# Assign normalised speed to agent properties before running dialogues.
		following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.agent_id, explanation_type="compact")
		if not following_regulation:
			return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation', explanation_list))
		#######################################
		# "Move forward" rule
		return null_reward(is_terminal=False, label='moving_forward')

class MultiAgentGraphDrive(MultiAgentEnv):
	metadata = {'render.modes': ['human', 'rgb_array']}
	
	def seed(self, seed=None):
		for i,a in enumerate(self.agent_list):
			seed = a.seed(seed+i)[0]
		self._seed = seed-1
		self.np_random, _ = seeding.np_random(self._seed)
		return [self._seed]

	def __init__(self, config=None):
		self.env_config = config
		self.num_agents = config.get('num_agents',1)
		self.viewer = None

		self.env_config['max_steering_angle'] = np.deg2rad(self.env_config['max_steering_degree'])
		self.env_config['max_steering_noise_angle'] = np.deg2rad(self.env_config['max_steering_noise_degree'])
		self.env_config['map_size'] = (self.env_config['max_dimension'], self.env_config['max_dimension'])
		self.env_config['min_junction_distance'] = 2.5*self.env_config['junction_radius']

		assert self.env_config['min_junction_distance'] > 2*self.env_config['junction_radius'], f"min_junction_distance has to be greater than {2*self.env_config['junction_radius']} but it is {self.env_config['min_junction_distance']}"
		assert self.env_config['max_speed']*self.env_config['mean_seconds_per_step'] < self.env_config['min_junction_distance'], f"max_speed*mean_seconds_per_step has to be lower than {self.env_config['min_junction_distance']} but it is {self.env_config['max_speed']*self.env_config['mean_seconds_per_step']}"

		logger.warning(f'Setting environment with reward_fn <{self.env_config["reward_fn"]}> and culture_level <{self.env_config["culture_level"]}>')
		self.culture = eval(f'{self.env_config["culture_level"]}RoadCulture')(
			road_options={
				'motorway': 1/2,
				'stop_sign': 1/2,
				'school': 1/2,
				'single_lane': 1/2,
				'town_road': 1/2,
				'roadworks': 1/8,
				'accident': 1/8,
				'heavy_rain': 1/2,
				'congestion_charge': 1/8,
			}, agent_options={
				'emergency_vehicle': 1/5,
				'heavy_vehicle': 1/4,
				'worker_vehicle': 1/3,
				'tasked': 1/2,
				'paid_charge': 1/2,
				'speed': self.env_config['max_normalised_speed'],
			}
		)

		self.agent_list = [
			GraphDriveAgent(self.num_agents-1, self.culture, self.env_config)
			for _ in range(self.num_agents)
		]
		self.action_space = self.agent_list[0].action_space
		self.observation_space = self.agent_list[0].observation_space
		self._agent_ids = set(range(self.num_agents))
		self.seed(config.get('seed',42))

	def reset(self):
		self.culture.np_random = self.np_random
		self.culture.seed = self._seed
		###########################
		self.road_network = MultiAgentRoadNetwork(
			self.culture, 
			map_size=self.env_config['map_size'], 
			min_junction_distance=self.env_config['min_junction_distance'],
			max_roads_per_junction=self.env_config['max_roads_per_junction'],
			number_of_agents=self.num_agents,
		)
		self.road_network.set(self.env_config['junction_number'])
		starting_point_list = self.road_network.get_random_starting_point_list(n=self.num_agents)
		for uid,agent in enumerate(self.agent_list):
			agent.reset(
				starting_point_list[uid], 
				self.road_network.agent_list[uid], 
				self.road_network, 
				self.agent_list[:uid]+self.agent_list[uid+1:]
			)
		# get_state is gonna use information about all agents, so initialize them first
		initial_state_dict = {
			uid: agent.get_state()
			for uid,agent in enumerate(self.agent_list)
		}
		return initial_state_dict

	def step(self, action_dict):
		state_dict, reward_dict, terminal_dict, info_dict = {}, {}, {}, {}
		step_info_dict = {
			uid: self.agent_list[uid].start_step(action)
			for uid,action in action_dict.items()
		}
		# end_step uses information about all agents, this requires all agents to act first and compute rewards and states after everybody acted
		for uid,step_info in step_info_dict.items():
			state_dict[uid], reward_dict[uid], terminal_dict[uid], info_dict[uid] = self.agent_list[uid].end_step(*step_info)
		terminal_dict['__all__'] = all(terminal_dict.values())
		return state_dict, reward_dict, terminal_dict, info_dict
			
	def get_info(self):
		return json.dumps({
			uid: agent.get_info()
			for uid,agent in enumerate(self.agent_list)
		}, indent=4)
		
	def get_screen(self): # RGB array
		# First set up the figure and the axis
		# fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, sharey=False, sharex=False, figsize=(10,10)) # this method causes memory leaks
		figure = Figure(figsize=(5,5), tight_layout=True)
		canvas = FigureCanvas(figure)
		ax = figure.add_subplot(111) # nrows=1, ncols=1, index=1
		
		# [Junctions]
		if len(self.road_network.junctions) > 0:
			junctions = [
				Circle(
					junction.pos, 
					self.env_config['junction_radius'], 
					color='y' if get_normalized_visit_count(junction, self.env_config['max_visit_per_junction']) < 1 else 'r', 
					alpha=0.25,
					# label=f"{get_normalized_visit_count(junction, self.env_config['max_visit_per_junction']):.2f}"
				)
				for junction in self.road_network.junctions
			]
			patch_collection = PatchCollection(junctions, match_original=True)
			ax.add_collection(patch_collection)

		# [Car]
		for uid,agent in enumerate(self.agent_list):
			car_x, car_y = agent.car_point
			car_handle = ax.scatter(car_x, car_y, marker='o', color='g' if not agent.slowed_down else 'r', label='Car')
			# [Heading Vector]
			dir_x, dir_y = get_heading_vector(angle=agent.car_orientation, space=self.env_config['max_dimension']/16)
			heading_vector_handle, = ax.plot([car_x, car_x+dir_x],[car_y, car_y+dir_y], color='g', alpha=0.5, label='Heading Vector')

		# [Roads]
		for road in self.road_network.roads:
			road_pos = list(zip(*(road.start.pos, road.end.pos)))
			road_colour = "Green" if road.colour is None else road.colour
			line_style = '-'
			path_handle, = ax.plot(road_pos[0], road_pos[1], color=colour_to_hex(road_colour), ls=line_style, lw=2, alpha=0.5, label="Road")

		# path1_handle, = ax.plot((0,0), (0,0), color=colour_to_hex("Green"), lw=2, label="OK")
		# path2_handle, = ax.plot((0,0), (0,0), color=colour_to_hex("Red"), lw=2, label="Unfeasible")
		# path3_handle, = ax.plot((0,0), (0,0), color=colour_to_hex("Gold"), lw=2, label="Wrong Speed")
		# path4_handle, = ax.plot((0,0), (0,0), color=colour_to_hex("Black"), ls='--', lw=2, label="Closest Road")
		# path5_handle, = ax.plot((0,0), (0,0), color=colour_to_hex("Black"), ls='-.', lw=2, label="Old Road")
		# junction_handle = ax.scatter(0, 0, marker='o', color='y', label='Junction')

		# Adjust ax limits in order to get the same scale factor on both x and y
		a,b = ax.get_xlim()
		c,d = ax.get_ylim()
		max_length = max(d-c, b-a)
		ax.set_xlim([a,a+max_length])
		ax.set_ylim([c,c+max_length])
		# # Build legend
		# handles = [car_handle, path1_handle, path2_handle, path3_handle, path4_handle, path5_handle]
		# ax.legend(handles=handles)
		# Draw plot
		# figure.suptitle(' '.join([
		# 	# f'[Angle]{np.rad2deg(self.steering_angle):.2f}°', 
		# 	# f'[Orient.]{np.rad2deg(self.car_orientation):.2f}°', 
		# 	# f'[Speed]{self.speed:.2f} m/s', 
		# 	# '\n',
		# 	f'[Step]{self._step}', 
		# 	# f'[Old]{self.closest_road.is_visited_by(self.agent_id)}', 
		# 	# f'[Car]{self.agent_id.binary_features()}', 
		# 	# f'[Reward]{self.last_reward:.2f}',
		# ]))
		# figure.tight_layout()
		canvas.draw()
		# Save plot into RGB array
		data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
		figure.clear()
		return data # RGB array

	def render(self, mode='human'):
		img = self.get_screen()
		if mode == 'rgb_array':
			return img
		elif mode == 'human':
			from gym.envs.classic_control import rendering
			if self.viewer is None:
				self.viewer = rendering.SimpleImageViewer()
			self.viewer.imshow(img)
			return self.viewer.isopen

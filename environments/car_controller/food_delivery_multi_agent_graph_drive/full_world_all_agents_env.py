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
from matplotlib.text import Text
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

from environments.car_controller.utils.geometry import *
from environments.car_controller.food_delivery_multi_agent_graph_drive.lib.multi_agent_road_network import MultiAgentRoadNetwork
from environments.car_controller.grid_drive.lib.road_cultures import *

import logging
logger = logging.getLogger(__name__)

# import time

normalize_food_count = lambda value, max_value: np.clip(value, 0, max_value)/max_value
is_source_junction = lambda j: j.is_available_source
is_target_junction = lambda j: j.is_available_target
EMPTY_FEATURE_PLACEHOLDER = -1

class FullWorldAllAgents_Agent:

	def seed(self, seed=None):
		# logger.warning(f"Setting random seed to: {seed}")
		self.np_random, _ = seeding.np_random(seed)
		return [seed]

	def __init__(self, n_of_other_agents, culture, env_config):
		# super().__init__()
		
		self.culture = culture
		self.n_of_other_agents = n_of_other_agents
		self.env_config = env_config
		self.max_relative_coordinates = 2*np.array(self.env_config['map_size'], dtype=np.float32)
		self.reward_fn = eval(f'self.{self.env_config["reward_fn"]}')
		self.fairness_reward_fn = eval(f'self.{self.env_config["fairness_reward_fn"]}') if self.env_config.get("fairness_reward_fn",None) else lambda x: 0
		
		self.obs_road_features = len(culture.properties) if culture else 0  # Number of binary ROAD features in Hard Culture
		self.obs_car_features = (len(culture.agent_properties) - 1) if culture else 0  # Number of binary CAR features in Hard Culture (excluded speed)
		# Spaces
		self.discrete_action_space = self.env_config['discrete_action_space']
		self.decides_acceleration = not self.env_config['force_car_to_stay_on_road'] or self.culture
		if self.discrete_action_space:
			self.allowed_steering_angles = np.linspace(-1, 1, self.env_config['n_discrete_actions']).tolist()
			if not self.decides_acceleration:
				self.allowed_accelerations = [1]
			else:
				self.allowed_accelerations = np.linspace(-1, 1, self.env_config['n_discrete_actions']).tolist()
			self.action_space = gym.spaces.Discrete(len(self.allowed_steering_angles)*len(self.allowed_accelerations))
		else:
			if self.decides_acceleration:
				self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1+1,), dtype=np.float32)  # steering angle and speed
			else:
				self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)  # steering angle
		state_dict = {
			"fc_junctions-16": gym.spaces.Box( # Junction properties and roads'
				low= -1,
				high= 1,
				shape= (
					self.env_config['junctions_number'],
					2 + 1 + 1 + 1 + 1, # junction.pos + junction.is_target + junction.is_source + junction.normalized_target_food + junction.normalized_source_food
				),
				dtype=np.float32
			),
			"fc_roads-16": gym.spaces.Box( # Junction properties and roads'
				low= -1,
				high= 1,
				shape= (
					self.env_config['junctions_number'],
					self.env_config['max_roads_per_junction'],
					2 + self.obs_road_features, # road.end + road.af_features
				),
				dtype=np.float32
			),
			"fc_this_agent-8": gym.spaces.Box( # Agent features
				low= -1,
				high= 1,
				shape= (
					self.agent_state_size,
				),
				dtype=np.float32
			),
		}
		if self.n_of_other_agents > 0:
			state_dict["fc_other_agents-16"] = gym.spaces.Box( # permutation invariant
				low= -1,
				high= 1,
				shape= (
					self.n_of_other_agents,
					2 + 1 + self.agent_state_size,
				), # for each other possible agent give position + orientation + state + features
				dtype=np.float32
			)
		self.observation_space = gym.spaces.Dict(state_dict)

		self._empty_junction = np.full(self.observation_space['fc_junctions-16'].shape[1:], EMPTY_FEATURE_PLACEHOLDER, dtype=np.float32)
		self._empty_road = np.full(self.observation_space['fc_roads-16'].shape[-1], EMPTY_FEATURE_PLACEHOLDER, dtype=np.float32)
		self._empty_junction_roads = np.full(self.observation_space['fc_roads-16'].shape[1:], EMPTY_FEATURE_PLACEHOLDER, dtype=np.float32)
		if self.n_of_other_agents > 0:
			self._empty_agent = np.full(self.observation_space['fc_other_agents-16'].shape[1:], EMPTY_FEATURE_PLACEHOLDER, dtype=np.float32)

	@property
	def step_gain(self):
		return 1/max(1,np.log(self.step)) # in (0,1]

	def reset(self, car_point, agent_id, road_network, other_agent_list):
		self.agent_id = agent_id
		self.road_network = road_network
		self.other_agent_list = other_agent_list
		self.seconds_per_step = self.get_step_seconds()
		self.slowdown_factor = self.get_slowdown_factor()
		# car position
		self.car_point = car_point
		self.car_orientation = np.mod(self.np_random.random()*two_pi, two_pi) # in [0,2*pi)
		self.distance_to_closest_road, self.closest_road, self.closest_junction_list = self.road_network.get_closest_road_and_junctions(self.car_point)
		self.closest_junction = MultiAgentRoadNetwork.get_closest_junction(self.closest_junction_list, self.car_point)
		# steering angle & speed
		self.steering_angle = 0
		self.speed = self.env_config['min_speed'] #+ (self.env_config['max_speed']-self.env_config['min_speed'])*self.np_random.random() # in [min_speed,max_speed]
		# self.speed = self.env_config['min_speed']+(self.env_config['max_speed']-self.env_config['min_speed'])*(70/120) # for testing
		if self.culture:
			self.agent_id.assign_property_value("Speed", self.road_network.normalise_speed(self.env_config['min_speed'], self.env_config['max_speed'], self.speed))

		self.last_closest_road = None
		self.last_closest_junction = None
		self.source_junction = None
		self.goal_junction = None
		# init concat variables
		self.last_reward = 0
		self.last_reward_type = 'move_forward'
		self.last_action_mask = None
		self.is_dead = False
		self.has_food = True
		# self.steps_in_junction = 0
		self.step = 1
		# self.idle = False

		self.visiting_new_road = False
		self.visiting_new_junction = False
		self.has_just_taken_food = False
		self.has_just_delivered_food = False

	def get_state(self, car_point=None, car_orientation=None):
		if car_point is None:
			car_point=self.car_point
		if car_orientation is None:
			car_orientation=self.car_orientation
		
		sorted_junctions = self.get_visible_junctions(car_point, car_orientation)
		junctions_view_list = self.get_junction_view_list(sorted_junctions, car_point, car_orientation, self.env_config['junctions_number'])
		roads_view_list = self.get_roads_view_list(sorted_junctions, car_point, car_orientation, self.env_config['junctions_number'])
		agent_feature_list = self.get_agent_feature_list()
		state_dict = {
			"fc_junctions-16": np.array(junctions_view_list, dtype=np.float32),
			"fc_roads-16": np.array(roads_view_list, dtype=np.float32),
			"fc_this_agent-8": np.array(agent_feature_list, dtype=np.float32),
		}
		if self.n_of_other_agents > 0:
			agent_neighbourhood_view = self.get_neighbourhood_view(car_point, car_orientation)
			state_dict["fc_other_agents-16"] = np.array(agent_neighbourhood_view, dtype=np.float32)
		return state_dict

	@property
	def agent_state_size(self):
		agent_state_size = 6
		if not self.env_config['force_car_to_stay_on_road']:
			agent_state_size += 1
		if self.env_config['blockage_probability']:
			agent_state_size += 1
		if self.culture:
			agent_state_size += self.obs_car_features
		return agent_state_size

	def get_agent_feature_list(self):
		agent_state = [
			# self.car_orientation/two_pi, # in [0,1) # needed in multi-agent environments, otherwise relative positions would be meaningless
			self.steering_angle/self.env_config['max_steering_angle'], # normalised steering angle # in [0,1]
			self.speed/self.env_config['max_speed'], # normalised speed # in [0,1]
			self.is_in_junction(self.car_point),
			self.has_food,
			self.is_dead,
			self.step_gain, # in (0,1]
		]
		if not self.env_config['force_car_to_stay_on_road']:
			agent_state.append(min(1, self.distance_to_closest_road/self.env_config['max_distance_to_path']))
		if self.env_config['blockage_probability']:
			agent_state.append(self.slowdown_factor)
		if self.culture:
			agent_state += self.agent_id.binary_features(as_tuple=True)
		return agent_state

	def colliding_with_other_agent(self, old_car_point, car_point):
		if not self.env_config['agent_collision_radius']:
			return False
		for agent in self.other_agent_list:
			if segment_collide_circle(segment=(old_car_point, car_point), circle=(agent.car_point,self.env_config['agent_collision_radius'])):
				return True
		return False

	def get_junction_roads(self, j, source_point, source_orientation):
		relative_road_pos_vector = shift_and_rotate_vector(
			[
				road.start.pos if j.pos!=road.start.pos else road.end.pos
				for road in j.roads_connected
			], 
			source_point, 
			source_orientation
		)
		relative_road_pos_vector /= self.max_relative_coordinates
		if self.culture:
			road_feature_vector = np.array(
				[
					road.binary_features(as_tuple=True) # in [0,1]
					for road in j.roads_connected
				], 
				dtype=np.float32
			)
			junction_road_list = np.concatenate(
				[
					relative_road_pos_vector,
					road_feature_vector
				], 
				axis=-1
			).tolist()
		else:
			junction_road_list = relative_road_pos_vector.tolist()
		junction_road_list.sort(key=lambda x: x[:2])
		
		missing_roads = [self._empty_road]*(self.env_config['max_roads_per_junction']-len(j.roads_connected))
		return junction_road_list + missing_roads

	def get_roads_view_list(self, sorted_junctions, source_point, source_orientation, n_junctions):
		return [
			self.get_junction_roads(sorted_junctions[i][1], source_point, source_orientation) 
			if i < len(sorted_junctions) else 
			self._empty_junction_roads
			for i in range(n_junctions)
		]

	def get_junction_view_list(self, sorted_junctions, source_point, source_orientation, n_junctions):
		return [
			np.array(
				(
					*sorted_junctions[i][0], 
					sorted_junctions[i][1].is_source, 
					normalize_food_count(
						sorted_junctions[i][1].food_refills, 
						self.env_config['max_food_per_source']
					) if sorted_junctions[i][1].is_source else -1,
					sorted_junctions[i][1].is_target, 
					normalize_food_count(
						sorted_junctions[i][1].food_deliveries, 
						self.env_config['max_food_per_target']
					) if sorted_junctions[i][1].is_target else -1,
				), 
				dtype=np.float32
			)
			if i < len(sorted_junctions) else 
			self._empty_junction
			for i in range(n_junctions)
		]

	def get_visible_junctions(self, source_point, source_orientation):
		relative_jpos_vector = shift_and_rotate_vector(
			[j.pos for j in self.road_network.junctions], 
			source_point, 
			source_orientation
		) / self.max_relative_coordinates
		sorted_junctions = sorted(zip(relative_jpos_vector.tolist(),self.road_network.junctions), key=lambda x: x[0])
		return sorted_junctions

	def get_neighbourhood_view(self, source_point, source_orientation):
		if self.other_agent_list:
			alive_agent = [x for x in self.other_agent_list if not x.is_dead]
			sorted_alive_agents = sorted((
				(
					shift_and_rotate_vector(agent.car_point, source_point, source_orientation) / self.max_relative_coordinates,
					agent.car_orientation/two_pi,
					agent.get_agent_feature_list(), 
				)
				for agent in alive_agent
			), key=lambda x: x[0])
			sorted_alive_agents = [
				(*agent_point, agent_orientation, *agent_state)
				for agent_point, agent_orientation, agent_state in sorted_alive_agents
			]
			agents_view_list = [
				np.array(sorted_alive_agents[i], dtype=np.float32) 
				if i < len(sorted_alive_agents) else 
				self._empty_agent
				for i in range(len(self.other_agent_list))
			]
		else:
			agents_view_list = None
		return agents_view_list

	def move(self, point, orientation, steering_angle, speed, add_noise=False):
		# https://towardsdatascience.com/how-self-driving-cars-steer-c8e4b5b55d7f?gi=90391432aad7
		# Add noise
		if add_noise:
			steering_angle += (2*self.np_random.random()-1)*self.env_config['max_steering_noise_angle']
			steering_angle = np.clip(steering_angle, -self.env_config['max_steering_angle'], self.env_config['max_steering_angle']) # |steering_angle| <= max_steering_angle, ALWAYS
			speed += (2*self.np_random.random()-1)*self.env_config['max_speed_noise']
		#### Ackerman Steering: Forward Kinematic for Car-Like vehicles #### https://www.xarg.org/book/kinematics/ackerman-steering/
		if steering_angle:
			turning_radius = self.env_config['wheelbase']/np.tan(steering_angle)
			# Max taylor approximation error of the tangent simplification is about 3° at 30° steering lock
			# turning_radius = self.env_config['wheelbase']/steering_angle
			angular_velocity = speed/turning_radius
			# get normalized new orientation
			new_orientation = np.mod(orientation + angular_velocity*self.seconds_per_step, two_pi) # in [0,2*pi)
		else:
			new_orientation = orientation
		# Move point
		x, y = point
		dir_x, dir_y = get_heading_vector(angle=new_orientation, space=speed*self.seconds_per_step)
		new_point = (x+dir_x, y+dir_y)
		return new_point, new_orientation

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

	def get_slowdown_factor(self):
		if not self.env_config['blockage_probability']:
			return 0
		if not self.np_random.choice(a=[False,True], p=[1-self.env_config['blockage_probability'],self.env_config['blockage_probability']]):
			return 0
		return self.np_random.uniform(self.env_config['min_blockage_ratio'], self.env_config['max_blockage_ratio'])

	def compute_distance_to_closest_road(self):
		if self.goal_junction is None:
			self.distance_to_closest_road, self.closest_road, self.closest_junction_list = self.road_network.get_closest_road_and_junctions(self.car_point, self.closest_junction_list)
		else:
			self.distance_to_closest_road = point_to_line_dist(self.car_point, self.closest_road.edge)

	def start_step(self, action_vector):
		if self.discrete_action_space:
			action_vector = [
				self.allowed_steering_angles[action_vector//len(self.allowed_accelerations)],
				self.allowed_accelerations[(action_vector%len(self.allowed_accelerations))]
			]
		# first of all, get the seconds passed from last step
		self.seconds_per_step = self.get_step_seconds()
		# compute new steering angle
		##################################
		# self.idle = False
		if self.env_config['optimal_steering_angle_on_road'] and not self.is_in_junction(self.car_point):
			road_edge = self.closest_road.edge if self.closest_road.edge[-1] == self.goal_junction.pos else self.closest_road.edge[::-1]
			if self.env_config['allow_uturns_on_edges']:
				if self.get_steering_angle_from_action(action=action_vector[0]) < 0:
					road_edge = road_edge[::-1]
					tmp = self.goal_junction
					self.goal_junction = self.source_junction
					self.source_junction = tmp
			# else:
			# 	self.idle = True
			self.car_orientation = get_slope_radians(*road_edge)%two_pi # in [0, 2*pi)
			self.steering_angle = 0
		else:
			self.steering_angle = self.get_steering_angle_from_action(action=action_vector[0])
		##################################
		# compute new acceleration and speed
		##################################
		self.speed = self.accelerate(
			speed=self.speed, 
			acceleration=self.get_acceleration_from_action(action=action_vector[1] if self.decides_acceleration else 1)
		)
		if self.culture:
			self.agent_id.assign_property_value("Speed", self.road_network.normalise_speed(self.env_config['min_speed'], self.env_config['max_speed'], self.speed))
		# move car
		old_car_point = self.car_point
		old_goal_junction = self.goal_junction
		self.visiting_new_road = False
		self.visiting_new_junction = False
		self.has_just_taken_food = False
		self.has_just_delivered_food = False

		self.car_point, self.car_orientation = self.move(
			point=self.car_point, 
			orientation=self.car_orientation, 
			steering_angle=self.steering_angle, 
			speed=self.speed*(1-self.slowdown_factor),
			add_noise=True
		)
		self.compute_distance_to_closest_road()
		if self.env_config['force_car_to_stay_on_road']:
			if not self.is_in_junction(self.car_point) and self.distance_to_closest_road >= self.env_config['max_distance_to_path']: # go back
				self.car_point = old_car_point
				self.speed = 0
				self.compute_distance_to_closest_road()
		self.closest_junction = MultiAgentRoadNetwork.get_closest_junction(self.closest_junction_list, self.car_point)
		# if a new road is visited, add the old one to the set of visited ones
		if self.is_in_junction(self.car_point):
			# self.steps_in_junction += 1
			if self.last_closest_road is not None: # if closest_road is not the first visited road
				self.closest_junction.is_visited_by(self.agent_id, True) # set the current junction as visited
				self.last_closest_road.is_visited_by(self.agent_id, True) # set the old road as visited
			if self.closest_junction != self.last_closest_junction:
				self.visiting_new_junction = True
				#########
				self.source_junction = None
				self.goal_junction = None
				self.last_closest_road = None
				self.last_closest_junction = self.closest_junction
				#########
				# if self.env_config['allow_uturns_on_edges']:
				if self.has_food:
					if is_target_junction(self.closest_junction) and self.road_network.deliver_food(self.closest_junction):
						self.has_food = False
						self.has_just_delivered_food = True
				else:
					if is_source_junction(self.closest_junction) and self.road_network.acquire_food(self.closest_junction):
						self.has_food = True
						self.has_just_taken_food = True
		else:
			if self.last_closest_road != self.closest_road: # not in junction and visiting a new road
				self.visiting_new_road = True
				#########
				self.last_closest_junction = None
				self.last_closest_road = self.closest_road # keep track of the current road
				self.goal_junction = MultiAgentRoadNetwork.get_furthermost_junction(self.closest_junction_list, self.car_point)
				self.source_junction = self.closest_junction
			# if not self.env_config['allow_uturns_on_edges']:
			# 	if self.has_food:
			# 		if is_target_junction(self.goal_junction) and self.road_network.deliver_food(self.goal_junction):
			# 			self.has_food = False
			# 			self.has_just_delivered_food = True
			# 	else:
			# 		if is_source_junction(self.goal_junction) and self.road_network.acquire_food(self.goal_junction):
			# 			self.has_food = True
			# 			self.has_just_taken_food = True
		return old_goal_junction, old_car_point

	def get_car_projection_on_road(self, car_point, closest_road):
		return poit_to_line_projection(car_point, closest_road.edge)

	def end_step(self, old_goal_junction, old_car_point):
		reward, dead, reward_type = self.reward_fn(old_goal_junction, old_car_point)
		how_fair = self.get_fairness_score()
		reward += self.fairness_reward_fn(how_fair)
		self.slowdown_factor = self.get_slowdown_factor()

		state = self.get_state()
		info_dict = {
			'explanation':{
				'why': reward_type,
				'how_fair': how_fair,
			},
			"stats_dict": {
				# "min_food_deliveries": self.road_network.min_food_deliveries,
				"food_deliveries": self.road_network.food_deliveries,
				"fair_food_deliveries": self.road_network.fair_food_deliveries,
				"food_refills": self.road_network.food_refills,
				# "avg_speed": (sum((x.speed for x in self.other_agent_list))+self.speed)/(len(self.other_agent_list)+1),
			},
			# 'discard': self.idle and not reward,
		}

		self.is_dead = dead
		self.step += 1
		self.last_reward = reward
		self.last_reward_type = reward_type
		return [state, reward, dead, info_dict]
			
	def get_info(self):
		return f"speed={self.speed}, steering_angle={self.steering_angle}, orientation={self.car_orientation}"

	def frequent_reward_default(self, old_goal_junction, old_car_point):
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def cost_reward(is_positive, is_terminal, label):
			r = self.step_gain # in (0,1]
			return (r if is_positive else -r, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		if not self.env_config['force_car_to_stay_on_road']:
			#######################################
			# "Is colliding" rule
			if self.colliding_with_other_agent(old_car_point, self.car_point):
				return unitary_reward(is_positive=False, is_terminal=True, label='has_collided_another_agent')

		#######################################
		# "Has delivered food to target" rule
		if self.has_just_delivered_food:
			return unitary_reward(is_positive=True, is_terminal=False, label='has_just_delivered_food_to_target')

		#######################################
		# "Has taken food from source" rule
		if self.has_just_taken_food:
			return unitary_reward(is_positive=True, is_terminal=False, label='has_just_taken_food_from_source')

		#######################################
		# "Is in junction" rule
		if self.is_in_junction(self.car_point):
			return null_reward(is_terminal=False, label='is_in_junction')

		if not self.env_config['force_car_to_stay_on_road']:
			#######################################
			# "Stay on the road" rule
			if self.distance_to_closest_road >= self.env_config['max_distance_to_path']:
				return unitary_reward(is_positive=False, is_terminal=True, label='not_staying_on_the_road')

		if self.culture:
			#######################################
			# "Follow regulation" rule. # Run dialogue against culture.
			# Assign normalised speed to agent properties before running dialogues.
			following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.agent_id, explanation_type="compact")
			if not following_regulation:
				return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation', explanation_list))

		#######################################
		# "Move forward" rule
		return null_reward(is_terminal=False, label='moving_forward')

	def sparse_reward_default(self, old_goal_junction, old_car_point):
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def cost_reward(is_positive, is_terminal, label):
			r = self.step_gain # in (0,1]
			return (r if is_positive else -r, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		#######################################
		# "Mission completed" rule
		if self.road_network.min_food_deliveries == self.env_config['max_food_per_target']:
			return cost_reward(is_positive=True, is_terminal=True, label='mission_completed')

		if not self.env_config['force_car_to_stay_on_road']:
			#######################################
			# "Is colliding" rule
			if self.colliding_with_other_agent(old_car_point, self.car_point):
				return unitary_reward(is_positive=False, is_terminal=True, label='has_collided_another_agent')

		#######################################
		# "Has delivered food to target" rule
		if self.has_just_delivered_food:
			return null_reward(is_terminal=False, label='has_just_delivered_food_to_target')

		#######################################
		# "Has taken food from source" rule
		if self.has_just_taken_food:
			return null_reward(is_terminal=False, label='has_just_taken_food_from_source')

		#######################################
		# "Is in junction" rule
		if self.is_in_junction(self.car_point):
			return null_reward(is_terminal=False, label='is_in_junction')

		if not self.env_config['force_car_to_stay_on_road']:
			#######################################
			# "Stay on the road" rule
			if self.distance_to_closest_road >= self.env_config['max_distance_to_path']:
				return unitary_reward(is_positive=False, is_terminal=True, label='not_staying_on_the_road')

		if self.culture:
			#######################################
			# "Follow regulation" rule. # Run dialogue against culture.
			# Assign normalised speed to agent properties before running dialogues.
			following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.agent_id, explanation_type="compact")
			if not following_regulation:
				return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation', explanation_list))
		
		#######################################
		# "Move forward" rule
		return null_reward(is_terminal=False, label='moving_forward')

	def get_fairness_score(self):
		####### Facts
		j = self.closest_junction #if self.env_config['allow_uturns_on_edges'] else self.goal_junction
		if self.has_just_delivered_food: 
			just_delivered_to_worst_target = j.food_deliveries == self.road_network.min_food_deliveries or j.food_deliveries-1 == self.road_network.min_food_deliveries
			return 'has_fairly_pursued_a_poor_target' if just_delivered_to_worst_target else 'has_pursued_a_rich_target'
		####### Conjectures
		elif self.visiting_new_junction:
			# is_exploring_fairly = not j.is_visited
			# if is_exploring_fairly:
			# 	return 'is_probably_exploring'
			if self.has_food:
				closest_target_type = self.road_network.get_closest_target_type(j, max_depth=3)
				if closest_target_type:
					if 'worst' in closest_target_type:
						return 'is_likely_to_fairly_pursue_a_poor_target_within_3_nodes'
					if closest_target_type=='best':
						return 'is_likely_to_pursue_a_rich_target_within_3_nodes'
		#######
		# if self.has_just_taken_food: 
		# 	return 'fair'
		# moving_towards_source_without_food = self.goal_junction and is_source_junction(self.goal_junction) and not self.has_food
		# if moving_towards_source_without_food:
		# 	return 'fair'
		#######
		return 'unknown'

	def sparse_fairness_reward(self, how_fair):
		return 1 if 'has_fairly_pursued_a_poor_target' == how_fair else 0

	def frequent_fairness_reward(self, how_fair):
		return 1 if 'fairly' in how_fair else 0


class FullWorldAllAgents_GraphDrive(MultiAgentEnv):
	metadata = {'render.modes': ['human', 'rgb_array']}
	
	def seed(self, seed=None):
		logger.warning(f"Setting random seed to: {seed}")
		for i,a in enumerate(self.agent_list):
			seed = a.seed(seed+i)[0]
		self._seed = seed-1
		self.np_random, _ = seeding.np_random(self._seed)
		# if self.culture:
		# 	self.culture.np_random = self.np_random
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

		logger.warning(f'Setting environment with reward_fn <{self.env_config["reward_fn"]}>, culture_level <{self.env_config["culture_level"]}> and fairness_reward_fn <{self.env_config["fairness_reward_fn"]}>')
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
		) if self.env_config["culture_level"] else None

		self.agent_list = [
			FullWorldAllAgents_Agent(self.num_agents-1, self.culture, self.env_config)
			for _ in range(self.num_agents)
		]
		self.action_space = self.agent_list[0].action_space
		self.observation_space = self.agent_list[0].observation_space
		self._agent_ids = set(range(self.num_agents))
		self.seed(config.get('seed',21))

	def reset(self):
		###########################
		self.road_network = MultiAgentRoadNetwork(
			self.culture, 
			self.np_random,
			map_size=self.env_config['map_size'], 
			min_junction_distance=self.env_config['min_junction_distance'],
			max_roads_per_junction=self.env_config['max_roads_per_junction'],
			number_of_agents=self.num_agents,
			junctions_number=self.env_config['junctions_number'],
			target_junctions_number=self.env_config['target_junctions_number'],
			source_junctions_number=self.env_config['source_junctions_number'],
			max_food_per_source=self.env_config['max_food_per_source'], 
			max_food_per_target=self.env_config['max_food_per_target'],
		)
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
		terminal_dict['__all__'] = all(terminal_dict.values()) or self.road_network.min_food_deliveries == self.env_config['max_food_per_target']
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

		def get_car_color(a):
			if a.is_dead:
				return 'grey'
			if a.slowdown_factor:
				return 'red'
			if a.has_food:
				return 'green'
			return colour_to_hex("Gold")
		car1_handle, = ax.plot((0,0), (0,0), color='green', lw=2, label="Has Food")
		car2_handle, = ax.plot((0,0), (0,0), color='red', lw=2, label="Is Slow")
		car3_handle, = ax.plot((0,0), (0,0), color='grey', lw=2, label="Is Dead")
		def get_junction_color(j):
			if is_target_junction(j):
				return 'red'
			if is_source_junction(j):
				return 'green'
			return 'grey'
		junction1_handle = ax.scatter(*self.road_network.target_junctions[0].pos, marker='o', color='red', alpha=0.5, label='Target Node')
		junction2_handle = ax.scatter(*self.road_network.source_junctions[0].pos, marker='o', color='green', alpha=0.5, label='Source Node')
		
		# [Car]
		#######################
		visibility_radius = self.env_config.get('visibility_radius',None)
		if visibility_radius: # [Visibility]
			visibility_view = [
				Circle(
					agent.car_point, 
					visibility_radius, 
					color='yellow', 
					alpha=0.25,
				)
				for uid,agent in enumerate(self.agent_list)
				if not agent.is_dead
			]
			ax.add_collection(PatchCollection(visibility_view, match_original=True))
		#######################
		car_view = [ # [Vehicle]
			Circle(
				agent.car_point, 
				1, 
				color=get_car_color(agent), 
				alpha=0.5,
			)
			for uid,agent in enumerate(self.agent_list)
		]
		ax.add_collection(PatchCollection(car_view, match_original=True))
		#######################
		for uid,agent in enumerate(self.agent_list): # [Rewards]
			if not agent.last_reward:
				continue
			ax.text(
				x=agent.car_point[0],
				y=agent.car_point[1]+0.5,
				s=f"{agent.last_reward:.2f}",
			)
		#######################
		for uid,agent in enumerate(self.agent_list): # [Debug info]
			if agent.visiting_new_road:
				ax.text(
					x=agent.car_point[0],
					y=agent.car_point[1],
					s='R', 
				)
			if agent.visiting_new_junction:
				ax.text(
					x=agent.car_point[0],
					y=agent.car_point[1],
					s='J', 
				)
		#######################
		for uid,agent in enumerate(self.agent_list): # [Heading Vector]
			car_x, car_y = agent.car_point
			dir_x, dir_y = get_heading_vector(angle=agent.car_orientation, space=self.env_config['max_dimension']/16)
			heading_vector_handle = ax.plot(
				[car_x, car_x+dir_x],[car_y, car_y+dir_y], 
				color=get_car_color(agent), 
				alpha=0.5, 
				# label='Heading Vector'
			)
		#######################
		# [Junctions]
		if len(self.road_network.junctions) > 0:
			junctions = [
				Circle(
					junction.pos, 
					self.env_config['junction_radius'], 
					color=get_junction_color(junction), 
					alpha=0.25,
					label='Target Node' if is_target_junction(junction) else ('Source Node' if is_source_junction(junction) else 'Normal Node')
				)
				for junction in self.road_network.junctions
			]
			patch_collection = PatchCollection(junctions, match_original=True)
			ax.add_collection(patch_collection)
			for junction in self.road_network.target_junctions:
				ax.annotate(
					junction.food_deliveries, 
					junction.pos, 
					color='black', 
					weight='bold', 
					fontsize=10, 
					ha='center', 
					va='center'
				)

		# [Roads]
		for road in self.road_network.roads:
			road_pos = list(zip(*(road.start.pos, road.end.pos)))
			line_style = '-'
			path_handle = ax.plot(road_pos[0], road_pos[1], color='black', ls=line_style, lw=2, alpha=0.5, label="Road")

		# Adjust ax limits in order to get the same scale factor on both x and y
		a,b = ax.get_xlim()
		c,d = ax.get_ylim()
		max_length = max(d-c, b-a)
		ax.set_xlim([a,a+max_length])
		ax.set_ylim([c,c+max_length])
		# Build legend
		handles = [car1_handle, car2_handle, car3_handle, junction1_handle, junction2_handle]
		ax.legend(handles=handles)
		# # Draw plot
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
			if self.viewer is None:
				from gym.envs.classic_control import rendering
				self.viewer = rendering.SimpleImageViewer()
			self.viewer.imshow(img)
			return self.viewer.isopen

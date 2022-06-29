# -*- coding: utf-8 -*-
import gym
from gym.utils import seeding
import numpy as np
import json
from more_itertools import unique_everseen
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
		self.discrete_action_space = self.env_config.get('n_discrete_actions',None)
		self.decides_speed = self.culture
		if self.discrete_action_space:
			self.allowed_orientations = np.linspace(-1, 1, self.env_config['n_discrete_actions']).tolist()
			if not self.decides_speed:
				self.allowed_speeds = [1]
			else:
				self.allowed_speeds = np.linspace(-1, 1, self.env_config['n_discrete_actions']).tolist()
			self.action_space = gym.spaces.Discrete(len(self.allowed_orientations)*len(self.allowed_speeds))
		else:
			if self.decides_speed:
				self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1+1,), dtype=np.float32)
			else:
				self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
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
				low= 0,
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

	def reset(self, car_point, agent_id, road_network, other_agent_list):
		self.agent_id = agent_id
		self.road_network = road_network
		self.other_agent_list = other_agent_list
		# car position
		self.car_point = car_point
		self.car_orientation = (self.np_random.random()*two_pi) % two_pi # in [0,2*pi)
		self.closest_junction = self.road_network.junction_dict[car_point]
		self.closest_road = None
		# speed
		self.car_speed = self.env_config['min_speed'] if self.decides_speed else self.env_config['max_speed']
		if self.culture:
			self.agent_id.assign_property_value("Speed", self.road_network.normalise_speed(self.env_config['min_speed'], self.env_config['max_speed'], self.car_speed))
		#####
		self.last_closest_road = None
		self.last_closest_junction = None
		self.source_junction = None
		self.goal_junction = None
		# init concat variables
		self.last_action_mask = None
		self.is_dead = False
		self.has_food = True
		# self.steps_in_junction = 0
		self.step = 1
		# self.idle = False
		self.last_reward = None

		self.visiting_new_road = False
		self.visiting_new_junction = False
		self.has_just_taken_food = False
		self.has_just_delivered_food = False

	@property
	def agent_state_size(self):
		agent_state_size = 4
		if self.decides_speed:
			agent_state_size += 1
		if self.culture:
			agent_state_size += self.obs_car_features
		return agent_state_size

	def get_agent_feature_list(self):
		agent_state = [
			self.is_in_junction(self.car_point),
			self.has_food,
			self.is_dead,
			self.step_gain, # in (0,1]
		]
		if self.decides_speed:
			agent_state.append(self.car_speed/self.env_config['max_speed']) # normalised speed # in [0,1]
		if self.culture:
			agent_state += self.agent_id.binary_features(as_tuple=True)
		return agent_state

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
	def step_gain(self):
		return 1/max(1,np.log(self.step)) # in (0,1]

	@property
	def step_seconds(self):
		return self.np_random.exponential(scale=self.env_config['mean_seconds_per_step']) if self.env_config['random_seconds_per_step'] else self.env_config['mean_seconds_per_step']
	
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
			sorted_alive_agents = sorted(
				(
					(
						(shift_and_rotate_vector(agent.car_point, source_point, source_orientation) / self.max_relative_coordinates).tolist(),
						agent.car_orientation/two_pi,
						agent.get_agent_feature_list(), 
					)
					for agent in alive_agent
				), 
				key=lambda x: x[0]
			)
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

	def is_in_junction(self, point, radius=None):
		if radius is None:
			radius = self.env_config['junction_radius']
		return euclidean_distance(self.closest_junction.pos, point) <= radius

	def is_on_road(self, point, max_distance=None):
		if max_distance is None:
			max_distance = self.env_config['max_distance_to_path']
		return point_to_line_dist(point, self.closest_road.edge) <= max_distance

	def move_car(self, max_space):
		x,y = self.car_point
		dx,dy = get_heading_vector(
			angle=self.car_orientation, 
			space=min(self.car_speed*self.step_seconds, max_space)
		)
		return (x+dx, y+dy)

	@property
	def neighbouring_junctions_iter(self):
		j_pos_set_iter = unique_everseen((
			j_pos
			for road in self.closest_junction.roads_connected 
			for j_pos in road.edge
		))
		return (
			self.road_network.junction_dict[j_pos]
			for j_pos in j_pos_set_iter
		)

	def start_step(self, action_vector):
		##################################
		## Get actions
		##################################
		if self.discrete_action_space:
			action_vector = (
				self.allowed_orientations[action_vector//len(self.allowed_speeds)],
				self.allowed_speeds[(action_vector%len(self.allowed_speeds))]
			)
		##################################
		## Compute new orientation
		##################################
		orientation_action = action_vector[0]
		# Optimal orientation on road
		if self.goal_junction: # is on road
			road_edge = self.closest_road.edge if self.closest_road.edge[-1] == self.goal_junction.pos else self.closest_road.edge[::-1]
			if self.env_config['allow_uturns_on_edges']:
				if orientation_action < 0: # invert direction; u-turn
					road_edge = road_edge[::-1]
					tmp = self.goal_junction
					self.goal_junction = self.source_junction
					self.source_junction = tmp
			self.car_orientation = get_slope_radians(*road_edge)%two_pi # in [0, 2*pi)
		else: # is in junction
			self.car_orientation = (self.car_orientation+(orientation_action+1)*pi)%two_pi
		##################################
		## Compute new speed
		##################################
		if self.decides_speed:
			speed_action = action_vector[1]
			self.car_speed = np.clip((speed_action+1)/2, self.env_config['min_speed'], self.env_config['max_speed'])
			if self.culture:
				self.agent_id.assign_property_value("Speed", self.road_network.normalise_speed(self.env_config['min_speed'], self.env_config['max_speed'], self.car_speed))
		##################################
		## Move car
		##################################
		distance_to_goal = euclidean_distance(self.car_point, self.goal_junction.pos) if self.goal_junction else float('inf')
		self.car_point = self.move_car(max_space=distance_to_goal)
		##################################
		## Get closest junction and road
		##################################
		old_closest_junction = self.closest_junction
		is_in_junction = self.is_in_junction(self.car_point) # This is correct because during reset cars are always spawn in a junction
		if not is_in_junction:
			if not self.closest_road:
				road_set = self.closest_junction.roads_connected if not self.env_config['random_seconds_per_step'] else unique_everseen((r for j in self.neighbouring_junctions_iter for r in j.roads_connected), key=lambda x:x.edge) # self.closest_junction.roads_connected is correct because we are asserting that self.env_config['max_speed']*self.env_config['mean_seconds_per_step'] < self.env_config['min_junction_distance']
				_,self.closest_road = self.road_network.get_closest_road_by_point(self.car_point, road_set)
		else:
			self.closest_road = None
		if self.closest_road:
			junction_set = (self.road_network.junction_dict[self.closest_road.edge[0]],self.road_network.junction_dict[self.closest_road.edge[1]])
			_,self.closest_junction = self.road_network.get_closest_junction_by_point(self.car_point, junction_set)
		# else:
		# 	_,self.closest_junction = self.road_network.get_closest_junction_by_point(self.car_point, self.neighbouring_junctions_iter)
		##################################
		## Adjust car position
		##################################
		# Force car to stay on a road or a junction
		if not is_in_junction and not self.is_on_road(self.car_point): # go back
			self.closest_junction = old_closest_junction
			self.closest_road = None
			self.car_point = self.closest_junction.pos
			is_in_junction = True
		##################################
		## Update the environment
		##################################
		self.visiting_new_road = False
		self.visiting_new_junction = False
		self.has_just_taken_food = False
		self.has_just_delivered_food = False
		if is_in_junction:
			# self.steps_in_junction += 1
			self.visiting_new_junction = self.closest_junction != self.last_closest_junction
			if self.visiting_new_junction: # visiting a new junction
				if self.last_closest_road is not None: # if closest_road is not the first visited road
					self.last_closest_road.is_visited_by(self.agent_id, True) # set the old road as visited
				self.closest_junction.is_visited_by(self.agent_id, True) # set the current junction as visited
				#########
				self.source_junction = None
				self.goal_junction = None
				self.last_closest_road = None
				self.last_closest_junction = self.closest_junction
				#########
				if self.has_food:
					if is_target_junction(self.closest_junction) and self.road_network.deliver_food(self.closest_junction):
						self.has_food = False
						self.has_just_delivered_food = True
				else:
					if is_source_junction(self.closest_junction) and self.road_network.acquire_food(self.closest_junction):
						self.has_food = True
						self.has_just_taken_food = True
		else:
			self.visiting_new_road = self.last_closest_road != self.closest_road		
			if self.visiting_new_road: # not in junction and visiting a new road
				self.last_closest_junction = None
				self.last_closest_road = self.closest_road # keep track of the current road
				self.goal_junction = self.road_network.junction_dict[self.closest_road.edge[0] if self.closest_road.edge[1] == self.closest_junction.pos else self.closest_road.edge[1]]
				self.source_junction = self.closest_junction

	def end_step(self):
		reward, dead, reward_type = self.reward_fn()
		how_fair = self.get_fairness_score()
		reward += self.fairness_reward_fn(how_fair)

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
				# "avg_speed": (sum((x.speed for x in self.other_agent_list))+self.car_speed)/(len(self.other_agent_list)+1),
			},
			# 'discard': self.idle and not reward,
		}

		self.is_dead = dead
		self.step += 1
		self.last_reward = reward
		return [state, reward, dead, info_dict]
			
	def get_info(self):
		return f"speed={self.car_speed}, orientation={self.car_orientation}"

	def frequent_reward_default(self):
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def cost_reward(is_positive, is_terminal, label):
			r = self.step_gain # in (0,1]
			return (r if is_positive else -r, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		#######################################
		# "Has delivered food to target" rule
		if self.has_just_delivered_food:
			return cost_reward(is_positive=True, is_terminal=False, label='has_just_delivered_food_to_target')

		#######################################
		# "Has taken food from source" rule
		if self.has_just_taken_food:
			# return cost_reward(is_positive=True, is_terminal=False, label='has_just_taken_food_from_source')
			return null_reward(is_terminal=False, label='has_just_taken_food_from_source')

		#######################################
		# "Is in junction" rule
		if self.is_in_junction(self.car_point):
			return null_reward(is_terminal=False, label='is_in_junction')

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

	def sparse_reward_default(self):
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
		# elif self.visiting_new_junction:
		# 	# is_exploring_fairly = not j.is_visited
		# 	# if is_exploring_fairly:
		# 	# 	return 'is_probably_exploring'
		# 	if self.has_food:
		# 		closest_target_type = self.road_network.get_closest_target_type(j, max_depth=3)
		# 		if closest_target_type:
		# 			if 'worst' in closest_target_type:
		# 				return 'is_likely_to_fairly_pursue_a_poor_target_within_3_nodes'
		# 			if closest_target_type=='best':
		# 				return 'is_likely_to_pursue_a_rich_target_within_3_nodes'
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
		for uid,action in action_dict.items():
			self.agent_list[uid].start_step(action)
		# end_step uses information about all agents, this requires all agents to act first and compute rewards and states after everybody acted
		state_dict, reward_dict, terminal_dict, info_dict = {}, {}, {}, {}
		for uid in action_dict.keys():
			state_dict[uid], reward_dict[uid], terminal_dict[uid], info_dict[uid] = self.agent_list[uid].end_step()
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
			if a.has_food:
				return 'green'
			return colour_to_hex("Gold")
		car1_handle, = ax.plot((0,0), (0,0), color='green', lw=2, label="Has Food")
		car3_handle, = ax.plot((0,0), (0,0), color='grey', lw=2, label="Is Dead")

		goal_junction_set = set((agent.goal_junction.pos for agent in self.agent_list if agent.goal_junction))
		def get_junction_color(j):
			if is_target_junction(j):
				return 'red'
			if is_source_junction(j):
				return 'green'
			if j.pos in goal_junction_set:
				return 'blue'
			# if j.pos in closest_junction_set:
			# 	return 'blue'
			return 'grey'
		junction1_handle = ax.scatter(*self.road_network.target_junctions[0].pos, marker='o', color='red', alpha=1, label='Target Node')
		junction2_handle = ax.scatter(*self.road_network.source_junctions[0].pos, marker='o', color='green', alpha=1, label='Source Node')
		if goal_junction_set:
			junction3_handle = ax.scatter(*next(iter(goal_junction_set)), marker='o', color='blue', alpha=1, label='Goal Node')
		
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
				alpha=1,
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
				alpha=1, 
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
					alpha=.5,
					label='Target Node' if is_target_junction(junction) else ('Source Node' if is_source_junction(junction) else 'Normal Node')
				)
				for junction in self.road_network.junctions
			]
			patch_collection = PatchCollection(junctions, match_original=True)
			ax.add_collection(patch_collection)
			for junction in self.road_network.target_junctions:
				ax.annotate(
					junction.food_deliveries, 
					(junction.pos[0],junction.pos[1]+0.5), 
					color='black', 
					# weight='bold', 
					fontsize=12, 
					ha='center', 
					va='center'
				)
			closest_junction_set = unique_everseen((agent.closest_junction for agent in self.agent_list), key=lambda x:x.pos)
			for junction in filter(lambda x: not x.is_target, closest_junction_set):
				ax.annotate(
					'#', 
					junction.pos, 
					color='black', 
					# weight='bold', 
					fontsize=12, 
					ha='center', 
					va='center'
				)

		# [Roads]
		closest_road_set = set((agent.closest_road.edge for agent in self.agent_list if agent.closest_road))
		for road in self.road_network.roads:
			road_pos = list(zip(*(road.start.pos, road.end.pos)))
			line_style = '--' if road.edge in closest_road_set else '-'
			path_handle = ax.plot(
				road_pos[0], road_pos[1], 
				color='black', 
				ls=line_style, 
				lw=2, 
				alpha=.5, 
				label="Road"
			)

		# Adjust ax limits in order to get the same scale factor on both x and y
		a,b = ax.get_xlim()
		c,d = ax.get_ylim()
		max_length = max(d-c, b-a)
		ax.set_xlim([a,a+max_length])
		ax.set_ylim([c,c+max_length])
		# Build legend
		handles = [car1_handle, car3_handle, junction1_handle, junction2_handle]
		if goal_junction_set:
			handles.append(junction3_handle)
		ax.legend(handles=handles)
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

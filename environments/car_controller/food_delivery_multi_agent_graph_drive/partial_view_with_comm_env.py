# -*- coding: utf-8 -*-
from environments.car_controller.food_delivery_multi_agent_graph_drive.global_view_env import *


class PartiallyObservableGraphDriveAgent(GraphDriveAgent):

	def __init__(self, n_of_other_agents, culture, env_config):
		super().__init__(n_of_other_agents, culture, env_config)
		self.max_n_junctions_in_view = int(np.ceil((self.env_config['visibility_radius']/self.env_config['min_junction_distance'])**2))
		# logger.warning(f'max_n_junctions_in_view: {self.max_n_junctions_in_view}')
		# min(self.env_config.get('max_n_junctions_in_view',float('inf')), self.env_config['junctions_number'])
		
		state_dict = {
			"fc_junctions-16": gym.spaces.Box( # Junction properties and roads'
				low= -1,
				high= 1,
				shape= (
					self.max_n_junctions_in_view,
					2 + 1 + 1 + 1, # junction.pos + junction.is_target + junction.is_source + junction.normalized_food_count
				),
				dtype=np.float32
			),
			"fc_roads-16": gym.spaces.Box( # Junction properties and roads'
				low= -1,
				high= 1,
				shape= (
					self.max_n_junctions_in_view,
					self.env_config['max_roads_per_junction'],
					2 + self.obs_road_features, # road.end + road.af_features
				),
				dtype=np.float32
			),
			"fc_this_agent-8": gym.spaces.Box( # Agent features
				low= -1,
				high= 1,
				shape= (
					self.agent_state_size + self.obs_car_features,
				),
				dtype=np.float32
			),
		}
		self.observation_space = gym.spaces.Dict(state_dict)
		# self._visible_road_network_junctions = None

	def get_state(self, car_point=None, car_orientation=None):
		if car_point is None:
			car_point=self.car_point
		if car_orientation is None:
			car_orientation=self.car_orientation
		junctions_view_list, roads_view_list = self.get_view(car_point, car_orientation)
		state_dict = {
			"fc_junctions-16": np.array(junctions_view_list, dtype=np.float32),
			"fc_roads-16": np.array(roads_view_list, dtype=np.float32),
			"fc_this_agent-8": np.array([
				*self.get_agent_state(),
				*(self.agent_id.binary_features(as_tuple=True) if self.culture else []), 
			], dtype=np.float32),
		}
		return state_dict

	# @property
	# def visible_road_network_junctions(self):
	# 	if self._visible_road_network_junctions is None or self.step % 16 == 0:
	# 		self._visible_road_network_junctions = list(filter(lambda j: self.can_see(j.pos), self.road_network.junctions))
	# 	return self._visible_road_network_junctions

	def can_see(self, p):
		return euclidean_distance(p, self.car_point) <= self.env_config['visibility_radius']

	def get_view(self, source_point, source_orientation): # source_orientation is in radians, source_point is in meters, source_position is quantity of past splines
		# s = time.time()
		source_x, source_y = source_point
		shift_rotate_normalise_point = lambda x: self.normalize_point(shift_and_rotate(*x, -source_x, -source_y, 0))
		visible_road_network_junctions = self.road_network.junctions
		visible_road_network_junctions = filter(lambda j: self.can_see(j.pos), visible_road_network_junctions) #self.visible_road_network_junctions
		visible_road_network_junctions = filter(lambda j: j.roads_connected, visible_road_network_junctions)
		visible_road_network_junctions = map(lambda j: {'junction_pos':shift_rotate_normalise_point(j.pos), 'junction':j}, visible_road_network_junctions)
		sorted_junctions = sorted(visible_road_network_junctions, key=lambda x: x['junction_pos'])

		##### Get junctions view
		junctions_view_list = [
			np.array(
				(
					*sorted_junctions[i]['junction_pos'], 
					sorted_junctions[i]['junction'].is_source, 
					sorted_junctions[i]['junction'].is_target, 
					get_normalized_food_count(sorted_junctions[i]['junction'],self.env_config['max_food_per_target']) if sorted_junctions[i]['junction'].is_target else -1
				), 
				dtype=np.float32
			) 
			if i < len(sorted_junctions) else 
			self._empty_junction
			for i in range(self.max_n_junctions_in_view)
		]

		##### Get roads view
		roads_view_list = [
			np.array(self.get_junction_roads(sorted_junctions[i]['junction'], shift_rotate_normalise_point), dtype=np.float32) 
			if i < len(sorted_junctions) else 
			self._empty_junction_roads
			for i in range(self.max_n_junctions_in_view)
		]

		# print('seconds',time.time()-s)
		return junctions_view_list, roads_view_list


class PVCommMultiAgentGraphDrive(MultiAgentGraphDrive):

	def __init__(self, config=None):
		super().__init__(config)
		self.agent_list = [
			PartiallyObservableGraphDriveAgent(self.num_agents-1, self.culture, self.env_config)
			for _ in range(self.num_agents)
		]
		self.action_space = self.agent_list[0].action_space
		base_space = self.agent_list[0].observation_space
		self.observation_space = gym.spaces.Dict({
			'all_agents_absolute_position_list': gym.spaces.Tuple(
				[gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(2,), dtype=np.float32)]*self.num_agents
			),
			# 'all_agents_absolute_orientation_list': gym.spaces.Tuple(
			# 	[gym.spaces.Box(low=0, high=two_pi, shape=(1,), dtype=np.float32)]*self.num_agents
			# ),
			'all_agents_relative_features_list': gym.spaces.Tuple(
				[base_space]*self.num_agents
			),
			'this_agent_id_mask': gym.spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32),
		})
		self.invisible_position_vec = np.array((float('inf'),float('inf')), dtype=np.float32)
		self.empty_agent_features = self.get_empty_state_recursively(base_space)
		self.agent_id_mask_dict = {}
		self.seed(config.get('seed',21))

	@staticmethod
	def get_empty_state_recursively(_obs_space):
		if isinstance(_obs_space, gym.spaces.Dict):
			return {
				k: PVCommMultiAgentGraphDrive.get_empty_state_recursively(v)
				for k,v in _obs_space.spaces.items()
			}
		elif isinstance(_obs_space, gym.spaces.Tuple):
			return list(map(PVCommMultiAgentGraphDrive.get_empty_state_recursively, _obs_space.spaces))
		return np.zeros(_obs_space.shape, dtype=_obs_space.dtype)

	def get_relative_position(self, this_agent_id, that_agent_id):
		if this_agent_id==that_agent_id:
			return (0,0)
		this_agent = self.agent_list[this_agent_id]
		source_x, source_y = this_agent.car_point
		source_orientation = this_agent.car_orientation

		that_agent = self.agent_list[that_agent_id]
		return shift_and_rotate(*that_agent.car_point, -source_x, -source_y, -source_orientation)

	def is_alive_and_visible(self, this_agent_id, that_agent_id):
		this_agent = self.agent_list[this_agent_id]
		that_agent = self.agent_list[that_agent_id]
		return not that_agent.is_dead and this_agent.can_see(that_agent.car_point)

	def get_this_agent_id_mask(self, this_agent_id):
		agent_id_mask = self.agent_id_mask_dict.get(this_agent_id,None)
		if agent_id_mask is None:
			agent_id_mask = np.zeros((self.num_agents,), dtype=np.float32)
			agent_id_mask[this_agent_id] = 1
			self.agent_id_mask_dict[this_agent_id] = agent_id_mask
		return agent_id_mask

	def build_state_with_comm(self, state_dict):
		if not state_dict:
			return state_dict

		all_agents_absolute_position_list = [
			np.array(self.agent_list[that_agent_id].car_point, dtype=np.float32) if that_agent_id in state_dict else self.invisible_position_vec
			for that_agent_id in range(self.num_agents)
		]
		# all_agents_absolute_orientation_list = [
		# 	np.array(self.agent_list[that_agent_id].car_orientation, dtype=np.float32) if that_agent_id in state_dict else 0
		# 	for that_agent_id in range(self.num_agents)
		# ]
		all_agents_relative_features_list = [
			state_dict.get(that_agent_id,self.empty_agent_features)
			for that_agent_id in range(self.num_agents)
		]
		return {
			this_agent_id: {
				'all_agents_absolute_position_list': all_agents_absolute_position_list,
				# 'all_agents_absolute_orientation_list': all_agents_absolute_orientation_list,
				'all_agents_relative_features_list': all_agents_relative_features_list,
				'this_agent_id_mask': self.get_this_agent_id_mask(this_agent_id),
			}
			for this_agent_id in state_dict.keys()
		}
		# new_state_dict = {}
		# for this_agent_id in state_dict.keys():
		# 	all_agents_absolute_position_list = []
		# 	all_agents_relative_features_list = []
		# 	for that_agent_id in range(self.num_agents):
		# 		if that_agent_id==this_agent_id or self.is_alive_and_visible(this_agent_id, that_agent_id):
		# 			all_agents_absolute_position_list.append(np.array(self.get_relative_position(this_agent_id, that_agent_id), dtype=np.float32))
		# 			all_agents_relative_features_list.append(state_dict[that_agent_id])
		# 		else:
		# 			all_agents_absolute_position_list.append(self.invisible_position_vec)
		# 			all_agents_relative_features_list.append(self.empty_agent_features)
		# 	new_state_dict[this_agent_id] = {
		# 		'all_agents_absolute_position_list': all_agents_absolute_position_list,
		# 		'all_agents_relative_features_list': all_agents_relative_features_list,
		# 		'this_agent_id_mask': self.get_this_agent_id_mask(this_agent_id),
		# 	}
		# return new_state_dict

	def reset(self):
		return self.build_state_with_comm(super().reset())

	def step(self, action_dict):
		state_dict, reward_dict, terminal_dict, info_dict = super().step(action_dict)
		# assert not any(terminal_dict.values())
		return self.build_state_with_comm(state_dict), reward_dict, terminal_dict, info_dict

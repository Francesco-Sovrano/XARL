# -*- coding: utf-8 -*-
from environments.car_controller.multi_agent_graph_drive.global_view_env import *


class PartiallyObservableGraphDriveAgent(GraphDriveAgent):

	def __init__(self, n_of_other_agents, culture, env_config):
		super().__init__(n_of_other_agents, culture, env_config)
		
		state_dict = {
			"fc_junctions-16": gym.spaces.Tuple([ # Tuple is a permutation invariant net, Dict is a concat net
				gym.spaces.Box( # Junction properties and roads'
					low= -1,
					high= 1,
					shape= (
						# self.env_config['junctions_number'],
						2 + 1 + 1 + 1, # junction.pos + junction.is_target + junction.is_source + junction.normalized_food_count
					),
					dtype=np.float32
				)
				for i in range(self.env_config['junctions_number'])
			]),
			"fc_roads-16": gym.spaces.Tuple([ # Tuple is a permutation invariant net, Dict is a concat net
				gym.spaces.Box( # Junction properties and roads'
					low= -1,
					high= 1,
					shape= (
						# self.env_config['junctions_number'],
						self.env_config['max_roads_per_junction'],
						1 + self.obs_road_features, # road.normalised_slope + road.af_features
					),
					dtype=np.float32
				)
				for i in range(self.env_config['junctions_number'])
			]),
			"fc_this_agent-16": gym.spaces.Box( # Agent features
				low= -1,
				high= 1,
				shape= (
					self.agent_state_size + self.obs_car_features,
				),
				dtype='bool'
			),
		}
		self.observation_space = gym.spaces.Dict(state_dict)

	def get_state(self, car_point=None, car_orientation=None):
		if car_point is None:
			car_point=self.car_point
		if car_orientation is None:
			car_orientation=self.car_orientation
		junctions_view_list, roads_view_list = self.get_view(car_point, car_orientation)
		state_dict = {
			"fc_junctions-16": junctions_view_list,
			"fc_roads-16": roads_view_list,
			"fc_this_agent-16": np.array([
				*self.get_agent_state(),
				*(self.agent_id.binary_features(as_tuple=True) if self.env_config["culture_level"] else []), 
			], dtype=np.float32),
		}
		return state_dict

	def can_see(self, p):
		return euclidean_distance(p, self.car_point) <= self.env_config['visibility_radius']

	def get_view(self, source_point, source_orientation): # source_orientation is in radians, source_point is in meters, source_position is quantity of past splines
		# s = time.time()
		source_x, source_y = source_point
		shift_rotate_normalise_point = lambda x: self.normalize_point(shift_and_rotate(*x, -source_x, -source_y, -source_orientation))
		road_network_junctions = filter(lambda j: j.roads_connected, self.road_network.junctions)
		road_network_junctions = filter(lambda j: self.can_see(j.pos), road_network_junctions)
		road_network_junctions = map(lambda j: {'junction_pos':shift_rotate_normalise_point(j.pos), 'junction':j}, road_network_junctions)
		sorted_junctions = sorted(road_network_junctions, key=lambda x: x['junction_pos'])

		##### Get junctions view
		junctions_view_list = [
			np.array((
				*sorted_junctions[i]['junction_pos'], 
				sorted_junctions[i]['junction'].is_source, 
				sorted_junctions[i]['junction'].is_target, 
				get_normalized_food_count(sorted_junctions[i]['junction'],self.env_config['max_food_per_target']) if sorted_junctions[i]['junction'].is_target else -1
				), dtype=np.float32
			) 
			if i < len(sorted_junctions) else 
			self._empty_junction
			for i in range(len(self.road_network.junctions))
		]

		##### Get roads view
		roads_view_list = [
			np.array(self.get_junction_roads(sorted_junctions[i]['junction']), dtype=np.float32) 
			if i < len(sorted_junctions) else 
			self._empty_junction_roads
			for i in range(len(self.road_network.junctions))
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
			'this_agent': base_space,
			'all_agents': gym.spaces.Tuple([base_space]*self.num_agents),
			'message_visibility_mask': gym.spaces.Box(low=0, high=1, shape= (self.num_agents,), dtype=np.float32),
		})
		self.seed(config.get('seed',21))

	def build_state_with_comm(self, state_dict):
		if not state_dict:
			return state_dict
		empty_state = next(iter(state_dict.values()))
		return {
			this_agent_id: {
				'this_agent': this_state, 
				'all_agents': [
					state_dict.get(that_agent_id, empty_state)
					for that_agent_id, that_agent in enumerate(self.agent_list)
				],
				'message_visibility_mask': np.array([
					this_agent_id != that_agent_id and that_agent_id in state_dict and self.agent_list[this_agent_id].can_see(that_agent.car_point)
					for that_agent_id, that_agent in enumerate(self.agent_list)
				], dtype=np.float32)
			}
			for this_agent_id,this_state in state_dict.items()
		}

	def reset(self):
		return self.build_state_with_comm(super().reset())

	def step(self, action_dict):
		state_dict, reward_dict, terminal_dict, info_dict = super().step(action_dict)
		return self.build_state_with_comm(state_dict), reward_dict, terminal_dict, info_dict

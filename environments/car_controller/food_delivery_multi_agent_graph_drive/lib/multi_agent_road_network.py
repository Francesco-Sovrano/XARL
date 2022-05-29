from environments.car_controller.graph_drive.lib.roads import RoadNetwork
from ...grid_drive.lib.road_agent import RoadAgent

class MultiAgentRoadNetwork(RoadNetwork):

	def __init__(self, culture, np_random, map_size=(50, 50), min_junction_distance=None, max_roads_per_junction=8, number_of_agents=5, junctions_number=10, target_junctions_number=5, source_junctions_number=5):
		assert junctions_number-target_junctions_number-source_junctions_number >= 0
		super().__init__(culture, np_random, map_size=map_size, min_junction_distance=min_junction_distance, max_roads_per_junction=max_roads_per_junction)
		### Agent
		del self.agent
		self.agent_list = [
			RoadAgent()
			for _ in range(number_of_agents)
		]
		if culture:
			for agent in self.agent_list:
				agent.set_culture(culture)
				culture.initialise_random_agent(agent, self.np_random)
		### Junction
		self.set(junctions_number)
		for j in self.junctions:
			j.is_source=False
			j.is_target=False
		self.target_junctions = []
		for j in self.np_random.choice(self.junctions, size=target_junctions_number, replace=False):
			j.is_target=True
			j.food_deliveries = 0
			self.target_junctions.append(j)
		non_target_junctions = [x for x in self.junctions if not x.is_target]
		self.source_junctions = []
		for j in self.np_random.choice(non_target_junctions, size=source_junctions_number, replace=False):
			j.is_source=True
			self.source_junctions.append(j)
		### Deliveries
		self.min_food_deliveries = 0
		self.food_deliveries = 0
		self.junction_dict = {
			x.pos: x
			for x in self.junctions
		}

	def deliver_food(self, j):
		j.food_deliveries += 1
		self.food_deliveries += 1
		self.min_food_deliveries = min(map(lambda x: x.food_deliveries, self.target_junctions))

	def get_target_type(self, j, is_target_fn):
		if not is_target_fn(j):
			return None
		if j.food_deliveries == self.min_food_deliveries:
			return 'worst'
		return 'best'

	def get_closest_target_type(self, start_junction, max_depth=float('inf'), is_target_fn=None):
		if not is_target_fn:
			is_target_fn = lambda x: x.is_target
		target_type = self.get_target_type(start_junction, is_target_fn)
		if target_type:
			return target_type
		visited_junction_set = set()
		junction_list = [start_junction]
		depth = 1
		while junction_list and depth < max_depth:
			visited_junction_set.update(map(lambda x: x.pos, junction_list))
			other_junction_list = []
			for junction in junction_list:
				least_advantaged_target_list = []
				advantaged_target_list = []
				for road in junction.roads_connected:
					other_junction = self.junction_dict[road.end.pos if road.start.pos == junction.pos else road.start.pos]
					target_type = self.get_target_type(other_junction, is_target_fn)
					if target_type == 'worst':
						least_advantaged_target_list.append(other_junction)
					elif target_type == 'best':
						advantaged_target_list.append(other_junction)
					elif other_junction.pos not in visited_junction_set:
						other_junction_list.append(other_junction)
				if least_advantaged_target_list and advantaged_target_list:
					return 'best_n_worst'
				if least_advantaged_target_list:
					return 'worst'
				if advantaged_target_list:
					return 'best'
			junction_list = other_junction_list
			depth += 1
		return None

	def get_random_starting_point_list(self, n=1):
		return [
			j.pos
			# for j in self.road_culture.np_random.choice(self.junctions, size=n, replace=False)
			for j in self.np_random.choice(self.source_junctions, size=n)
		]
		

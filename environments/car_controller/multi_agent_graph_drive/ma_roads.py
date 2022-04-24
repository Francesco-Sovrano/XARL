from environments.car_controller.graph_drive.lib.roads import RoadNetwork
from ..grid_drive.lib.road_agent import RoadAgent

class MultiAgentRoadNetwork(RoadNetwork):

	def __init__(self, culture, map_size=(50, 50), min_junction_distance=None, max_roads_per_junction=8, number_of_agents=5, junctions_number=10, target_junctions_number=5, source_junctions_number=5):
		assert junctions_number-target_junctions_number-source_junctions_number >= 0
		super().__init__(culture, map_size=map_size, min_junction_distance=min_junction_distance, max_roads_per_junction=max_roads_per_junction)
		### Agent
		del self.agent
		self.agent_list = [
			RoadAgent()
			for _ in range(number_of_agents)
		]
		for agent in self.agent_list:
			agent.set_culture(culture)
			self.road_culture.initialise_random_agent(agent)
		### Junction
		self.set(junctions_number)
		for j in self.junctions:
			j.is_source=False
			j.is_target=False
		self.target_junctions = []
		for j in self.road_culture.np_random.choice(self.junctions, size=target_junctions_number, replace=False):
			j.is_target=True
			j.food_deliveries = 0
			self.target_junctions.append(j)
		non_target_junctions = [x for x in self.junctions if not x.is_target]
		self.source_junctions = []
		for j in self.road_culture.np_random.choice(non_target_junctions, size=source_junctions_number, replace=False):
			j.is_source=True
			self.source_junctions.append(j)
		### Deliveries
		self.food_deliveries_dict = {0:len(self.junctions)}
		self.min_food_deliveries = 0

	def deliver_food(self, j):
		j_food_deliveries = j.food_deliveries
		self.food_deliveries_dict[j_food_deliveries] -= 1
		if not j_food_deliveries+1 in self.food_deliveries_dict:
			self.food_deliveries_dict[j_food_deliveries+1] = 1
		else:
			self.food_deliveries_dict[j_food_deliveries+1] += 1
		self.min_food_deliveries = min(self.food_deliveries_dict.items(), key=lambda x: x[-1])[0]

		j.food_deliveries += 1

	def get_random_starting_point_list(self, n=1):
		return [
			j.pos
			# for j in self.road_culture.np_random.choice(self.junctions, size=n, replace=False)
			for j in self.road_culture.np_random.choice(self.source_junctions, size=n)
		]
		

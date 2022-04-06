from environments.car_controller.graph_drive.lib.roads import RoadNetwork
from ..grid_drive.lib.road_agent import RoadAgent

class MultiAgentRoadNetwork(RoadNetwork):

	def __init__(self, culture, map_size=(50, 50), min_junction_distance=None, max_roads_per_junction=8, number_of_agents=5):
		super().__init__(culture, map_size=map_size, min_junction_distance=min_junction_distance, max_roads_per_junction=max_roads_per_junction)
		del self.agent
		self.agent_list = [
			RoadAgent()
			for _ in range(number_of_agents)
		]
		for agent in self.agent_list:
			agent.set_culture(culture)
			self.road_culture.initialise_random_agent(agent)

	def get_random_starting_point_list(self, n=1):
		# print(self.junctions)
		return [
			j.pos
			for j in self.road_culture.np_random.choice(self.junctions, n)
		]
		

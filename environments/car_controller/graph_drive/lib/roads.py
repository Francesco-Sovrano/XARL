import numpy as np
from collections import deque
from ...grid_drive.lib.road_cultures import * # FIXME: Move RoadCultures to a more generic location.
from ...grid_drive.lib.road_cell import RoadCell
from ...grid_drive.lib.road_agent import RoadAgent
from environments.car_controller.utils import *
from environments.car_controller.graph_drive.lib.simple_culture import SimpleCulture
from environments.utils.random_planar_graph.GenerateGraph import get_random_planar_graph, default_seed
import random

class Junction:
	max_roads_connected = 8
	def __init__(self, pos):
		self.pos = pos
		self.roads_connected = []

	def __eq__(self, other):
		return self.pos == other.pos

	def can_connect(self):
		return len(self.roads_connected) < self.max_roads_connected

	def connect(self, road):
		if not self.can_connect():
			return False
		if road not in self.roads_connected:
			self.roads_connected.append(road)
		return True

class Road(RoadCell):
	def __init__(self, start: Junction, end: Junction, connect=False):
		# arbitrary sorting by x-coordinate to avoid mirrored duplicates
		super().__init__()
		if start.pos[0] < end.pos[0]:
			start, end = end, start
		self.orientation = np.arctan2(end.pos[1] - start.pos[1], end.pos[0] - start.pos[0]) # get slope
		self.start = start
		self.end = end
		self.edge = (start.pos, end.pos)
		self.is_connected = False
		if connect:
			if self.start.can_connect() and self.end.can_connect():
				self.connect_to_junctions()
				self.is_connected = True

	def __eq__(self, other):
		return self.start == other.start and self.end == other.end

	@property
	def id(self):
		return (self.start.pos, self.end.pos)

	def connect_to_junctions(self):
		self.start.connect(self)
		self.end.connect(self)

	def get_orientation_relative_to(self, source_orientation):
		# normalise orientations
		source_orientation %= two_pi
		road_a_orientation = self.orientation % two_pi
		road_b_orientation = (road_a_orientation+np.pi) % two_pi
		road_a_orientation_relative_to_source = get_orientation_of_a_relative_to_b(road_a_orientation,source_orientation)
		road_b_orientation_relative_to_source = get_orientation_of_a_relative_to_b(road_b_orientation,source_orientation)
		# roads are two-way, get the closest orientation to source_orientation
		if abs(road_a_orientation_relative_to_source) < abs(road_b_orientation_relative_to_source):
			return road_a_orientation_relative_to_source
		return road_b_orientation_relative_to_source

class RoadNetwork:

	def __init__(self, culture, map_size=(50, 50), max_road_length=15, min_junction_distance=None):
		self.junctions = []
		self.roads = []
		self.map_size = map_size
		self.max_road_length = max_road_length
		self.agent = RoadAgent()
		if min_junction_distance is None:
			self.min_junction_distance = map_size[0]/8
		else:
			self.min_junction_distance = min_junction_distance
		self.road_culture = culture
		self.agent.set_culture(culture)
		self.road_culture.initialise_random_agent(self.agent)

	def normalise_speed(self, min, max, current):
		"""
		Normalises speed from Euclidean m/s to nominal speeds used in the culture rules (0-100)
		Args:
			min: min speed in m/s
			max: max speed in m/s
			current: current speed in m/s
		Returns: speed normalised to range (0-120)
		"""
		return 120 * ((current - min) / (max - min))

	def add_junction(self, junction):
		if junction not in self.junctions:
			self.junctions.append(junction)

	def add_road(self, road):
		if road not in self.roads:
			self.roads.append(road)

	def get_visible_junctions_by_point(self, source_point, horizon_distance):
		return [
			junction
			for junction in self.junctions
			if euclidean_distance(source_point, junction.pos) <= horizon_distance
		]

	def get_closest_road_and_junctions(self, point, closest_junctions=None):
		# the following lines of code are correct because the graph is planar
		if not closest_junctions:
			distance_to_closest_road, closest_road = self.get_closest_road_by_point(point)
		else:
			distance_to_closest_road, closest_road = min(
				(
					(
						point_to_line_dist(point, r.edge),
						r
					)
					for j in closest_junctions
					for r in j.roads_connected
				), key=lambda x:x[0]
			)
		road_start, road_end = closest_road.edge
		closest_junctions = [self.junction_dict[road_start],self.junction_dict[road_end]]
		return distance_to_closest_road, closest_road, closest_junctions

	def get_closest_junction_by_point(self, source_point):
		return min(
			(
				(
					euclidean_distance(junction.pos,source_point),
					junction
				)
				for junction in self.junctions
			), key=lambda x:x[0]
		)
	
	def get_closest_road_by_point(self, source_point):
		return min(
			(
				(
					point_to_line_dist(source_point,road.edge),
					road
				)
				for road in self.roads
			), key=lambda x:x[0]
		)

	def set(self, nodes_amount):
		self.junctions = []
		self.roads = []
		random_planar_graph = get_random_planar_graph({
			"width": self.map_size[0], # "Width of the field on which to place points.  neato might choose a different width for the output image."
			"height": self.map_size[1], # "Height of the field on which to place points.  As above, neato might choose a different size."
			"nodes": nodes_amount, # "Number of nodes to place."
			"edges": 2*nodes_amount, # "Number of edges to use for connections.  Double edges aren't counted."
			"radius": self.map_size[0]/8, # "Nodes will not be placed within this distance of each other."
			"double": 0, # "Probability of an edge being doubled."
			"hair": 0, # "Adjustment factor to favour dead-end nodes.  Ranges from 0.00 (least hairy) to 1.00 (most hairy).  Some dead-ends may exist even with a low hair factor."
			"seed": default_seed(), # "Seed for the random number generator."
			"debug_trimode": 'conform', # ['pyhull', 'triangle', 'conform'], "Triangulation mode to generate the initial triangular graph.  Default is conform.")
			"debug_tris": None, # "If a filename is specified here, the initial triangular graph will be saved as a graph for inspection."
			"debug_span": None, # "If a filename is specified here, the spanning tree will be saved as a graph for inspection."
		})
		self.junction_dict = dict(zip(random_planar_graph['nodes'], map(Junction, random_planar_graph['nodes'])))
		self.junctions = tuple(self.junction_dict.values())
		spanning_tree_set = set(random_planar_graph['spanning_tree'])
		# print('edges', random_planar_graph['edges'])
		for edge in random_planar_graph['edges']:
			p1,p2 = edge
			j1 = self.junction_dict[p1]
			j2 = self.junction_dict[p2]
			road = Road(j1, j2, connect=True)
			road.set_culture(self.road_culture)
			self.road_culture.initialise_random_road(road)
			self.roads.append(road)
		starting_point = random.choice(random_planar_graph['spanning_tree'])[0]
		return starting_point

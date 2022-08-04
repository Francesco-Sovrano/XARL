from ....utils.culture_lib.culture import Culture, Argument
from ....utils.road_lib.road_cell import RoadCell
from ....utils.road_lib.road_agent import RoadAgent

class HeterogeneityCulture(Culture):
	starting_argument_id = 0

	def __init__(self, road_options=None, agent_options=None):
		if road_options is None: road_options = {}
		if agent_options is None: agent_options = {}
		self.road_options = road_options
		self.agent_options = agent_options
		self.ids = {}
		super().__init__()
		self.name = "Heterogeneity Culture"
		# Properties of the culture with their default values go in self.properties.
		self.properties = {
			"Require Priority": False,
			"Accident": False,
			"Require Fee": False
		}

		self.agent_properties = {
			"Emergency Vehicle": False,
			"Is Prioritised": False,
			"Can Pay Fee": False
		}

	def initialise_feasible_road(self, road: RoadCell):
		for p in self.properties.keys():
			road.assign_property_value(p, False)

	def run_default_dialogue(self, road, agent, explanation_type="verbose"):
		"""
		Runs dialogue to find out decision regarding penalty in argumentation framework.
		Args:
			road: RoadCell corresponding to destination cell.
			agent: RoadAgent corresponding to agent.
			explanation_type: 'verbose' for all arguments used in exchange; 'compact' for only winning ones.

		Returns: Decision on penalty + explanation.
		"""
		# Game starts with proponent using argument 0 ("I will not get a ticket").
		return super().run_dialogue(road, agent, starting_argument_id=self.starting_argument_id, explanation_type=explanation_type)

	def initialise_random_road(self, road: RoadCell, np_random):
		"""
		Receives an empty RoadCell and initialises properties with acceptable random values.
		:param road: uninitialised RoadCell.
		"""
		road.assign_property_value("Require Priority", np_random.random() <= self.road_options.get('require_priority',1/2))
		road.assign_property_value("Accident", np_random.random() <= self.road_options.get('accident',1/8))
		road.assign_property_value("Require Fee", np_random.random() <= self.road_options.get('require_fee',1/2))

	def initialise_random_agent(self, agent: RoadAgent, np_random):
		"""
		Receives an empty RoadAgent and initialises properties with acceptable random values.
		:param agent: uninitialised RoadAgent.
		"""
		agent.assign_property_value("Emergency Vehicle", np_random.random() <= self.agent_options.get('emergency_vehicle',1/5))
		agent.assign_property_value("Is Prioritised", np_random.random() <= self.agent_options.get('has_priority',1/2))
		agent.assign_property_value("Can Pay Fee", np_random.random() <= self.agent_options.get('can_pay_fee',1/2))
	
	def create_arguments(self):
		"""
		Defines set of arguments present in the culture and their verifier functions.
		"""
		def build_argument(_id, _label, _description, _verifier_fn):
			motion = Argument(_id, _description)
			self.ids[_label] = _id
			motion.set_verifier(_verifier_fn)  # Propositional arguments are always valid.
			return motion

		self.AF.add_arguments([
			build_argument(0, "ok", "Nothing wrong.", lambda *gen: True), # Propositional arguments are always valid.
			build_argument(1, "emergency_vehicle", "You are an emergency vehicle.", lambda road, agent: agent["Emergency Vehicle"] is True),
			build_argument(2, "is_prioritised", "You have priority.", lambda road, agent: agent["Is Prioritised"] is True),
			build_argument(3, "accident", "There is an accident ahead.", lambda road, agent: road["Accident"] is True),
			build_argument(4, "required_priority", "You drove into a road that requires priority.", lambda road, agent: road["Require Priority"] is True),
			build_argument(5, "required_fee", "You drove into a road that requires a fee.", lambda road, agent: road["Require Fee"] is True),
			build_argument(6, "can_pay_fee", "You can pay the fee.", lambda road, agent: agent["Can Pay Fee"] is True),
		])

	def define_attacks(self):
		"""
		Defines attack relationships present in the culture.
		Culture can be seen here:
		https://docs.google.com/document/d/1O7LCeRVVyCFnP-_8PVcfNrEdVEN5itGxcH1Ku6GN5MQ/edit?usp=sharing
		"""
		ID = self.ids

		# 1
		self.AF.add_attack(ID["required_fee"], ID["ok"])
		self.AF.add_attack(ID["emergency_vehicle"], ID["required_fee"])
		self.AF.add_attack(ID["can_pay_fee"], ID["required_fee"])

		# 2
		self.AF.add_attack(ID["required_priority"], ID["ok"])
		self.AF.add_attack(ID["is_prioritised"], ID["required_priority"])

		# 3
		self.AF.add_attack(ID["accident"], ID["ok"])
		self.AF.add_attack(ID["emergency_vehicle"], ID["accident"])


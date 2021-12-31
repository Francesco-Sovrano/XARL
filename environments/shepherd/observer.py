from collections import deque
from scipy.spatial import distance
import numpy as np

from game import ShepherdGame, Dog, Sheep, Pen
from explanation import Explanation


class ShepherdObserver:
    explainers = {}

    def __init__(self, game: ShepherdGame, max_observed_neighbours):
        self.game = game
        self.buffer_size = 2
        self.observation_buffer = {}  # {agent : obs_buffer}
        self.max_observed_neighbours = max_observed_neighbours

        self.coordinate_space = np.linspace(0, self.game.map_side-1, self.game.map_side, dtype=np.float32)

        for agent in self.game.dogs:
            self.observation_buffer[agent] = deque(maxlen=self.buffer_size)

    def update(self):
        saved, lost = self.game.count_sheep(count_and_remove=False)
        reward = saved - lost

        for agent in self.game.dogs:
            obs = {"agent_pos": np.copy(agent.pos), "pen_pos": np.copy(self.game.sheep_pen.pos), "saved": saved, "lost": lost}

            # Get neighbours.
            neighbour_objs = self.game.get_neighbours_of(agent, self.game.dog_sense_radius)
            # Remove duplicates.
            if self.game.sheep_pen in neighbour_objs:
                neighbour_objs.remove(self.game.sheep_pen)
            neighbour_objs.remove(agent)

            neighbour_data = []
            for n in neighbour_objs:
                neighbour_dict = {"type": type(n), "pos": np.copy(n.pos)}
                neighbour_data.append(neighbour_dict)

            obs["neighbours"] = neighbour_data
            self.observation_buffer[agent].append(obs)

    def prepare_obs_for_env(self, obs):

        new_obs = {}
        new_obs["agent_pos"] = obs["agent_pos"]
        new_obs["pen_pos"] = obs["pen_pos"]

        # Preparing local view
        local_dim = (self.game.dog_sense_radius * 2) + 1
        global_dim = self.game.map_side - 1
        global_grid = self.game.global_grid

        coord_space = np.linspace(0, self.game.map_side - 1, self.game.map_side, dtype=np.float32)
        x = np.searchsorted(coord_space, obs["agent_pos"][0])
        y = np.searchsorted(coord_space, obs["agent_pos"][1])

        excess_x = 0
        excess_y = 0
        margin = local_dim/2
        if x < margin or x > global_dim - margin:
            excess_x = x - margin
        if y < margin or y > global_dim - margin:
            excess_y = y - margin

        cx = x - excess_x
        cy = y - excess_y

        local_view = global_grid[cy - margin: cy + margin, cx - margin, cy + margin]
        new_obs["local_view"] = np.copy(local_view)

        return new_obs




    def separate_neighbours_by_type(self, neighbours):
        neighbour_dict = {}
        for neighbour in neighbours:
            if neighbour["type"] in neighbour_dict:
                neighbour_dict[neighbour["type"]].append(neighbour)
            else:
                neighbour_dict[neighbour["type"]] = [neighbour]
        return neighbour_dict

    def explain(self, agent):
        obs_buffer = self.observation_buffer[agent]

        if len(obs_buffer) < 2:
            return []
        prev_obs = obs_buffer[-2]
        curr_obs = obs_buffer[-1]

        explanations = []
        for explanation_type, explanation_list in Explanation.explainers.items():
            for explainer in explanation_list:
                if explainer(self, prev_obs, curr_obs):
                    explanations.append(explanation_type + ':' + explainer.__name__)
        return explanations

    def herded_sheep(self, obs):
        neighbours = self.separate_neighbours_by_type(obs["neighbours"])
        if Sheep not in neighbours:
            return []
        sheep = neighbours[Sheep]
        herding_sheep = []
        for s in sheep:
            if distance.euclidean(s["pos"], obs["agent_pos"]) < self.game.sheep_sense_radius:
                herding_sheep.append(s)
        return herding_sheep

    @Explanation('WHY')
    def herding(self, prev_obs, curr_obs):
        """
        This WHY explanation returns True if the agent is currently herding any sheep at all.
        """
        return True if len(self.herded_sheep(curr_obs)) > 0 else False

    @Explanation('WHY')
    def saved_sheep(self, prev_obs, curr_obs):
        """
        This WHY explanation returns True if the agent is currently herding any sheep
        that gets into the sheep pen.
        """
        # Find sheep being herded.
        herding_sheep = self.herded_sheep(curr_obs)
        pen_pos = curr_obs["pen_pos"]
        sheep_pen = Pen(x=pen_pos[0], y=pen_pos[1], radius=self.game.pen_radius)
        # If there's a sheep that got into the pen, return True.
        for s in herding_sheep:
            sheep = Sheep(x=s["pos"][0], y=s["pos"][1], radius=self.game.sheep_size)
            if sheep_pen.contains(sheep):
                return True
        return False

    @Explanation('WHY')
    def lost_sheep(self, prev_obs, curr_obs):
        """
        This WHY explanation returns True if the agent is currently herding any sheep
        that gets out of the safe zone.
        """
        # Preliminary checks to avoid extra computation.
        agent_distance_to_centre = distance.euclidean(curr_obs["agent_pos"], self.game.map_centre)
        if agent_distance_to_centre < (self.game.map_side / 2.0) - self.game.sheep_sense_radius:
            return False
        # Find sheep being herded.
        herding_sheep = self.herded_sheep(curr_obs)
        # If there's a sheep that got out of the safe zone, return True.
        for s in herding_sheep:
            distance_to_centre = distance.euclidean(s["pos"], self.game.map_centre)
            if distance_to_centre > (self.game.map_side / 2.0):
                return True
        return False

    @Explanation('WHY')
    def moving_towards_pen(self, prev_obs, curr_obs):
        """
        This WHY explanation returns True if the agent is moving closer to the pen.
        """
        prev_distance = distance.euclidean(prev_obs["agent_pos"], prev_obs["pen_pos"])
        curr_distance = distance.euclidean(curr_obs["agent_pos"], curr_obs["pen_pos"])
        return curr_distance < prev_distance

    @Explanation('WHY')
    def not_moving(self, prev_obs, curr_obs):
        """
        This WHY explanation returns True if the agent is not moving.
        """
        prev_distance = distance.euclidean(prev_obs["agent_pos"], prev_obs["pen_pos"])
        curr_distance = distance.euclidean(curr_obs["agent_pos"], curr_obs["pen_pos"])
        return abs(curr_distance - prev_distance) < 0.01

    @Explanation('WHY')
    def someone_else_saved_sheep(self, prev_obs, curr_obs):
        """
        This WHY explanation returns True if the agent receives a positive reward
        but did not push any sheep into a pen.
        """
        return curr_obs["saved"] > 0 and not self.saved_sheep(prev_obs, curr_obs)

    @Explanation('WHY')
    def someone_else_lost_sheep(self, prev_obs, curr_obs):
        """
        This WHY explanation returns True if the agent receives a positive reward
        but did not push any sheep into a pen.
        """
        return curr_obs["lost"] > 0 and not self.lost_sheep(prev_obs, curr_obs)

    @Explanation('WHAT')
    def seeing_pen(self, prev_obs, curr_obs):
        """
        This WHAT explanation returns True if the pen is in the agent's view.
        """
        curr_distance = distance.euclidean(curr_obs["agent_pos"], curr_obs["pen_pos"])
        return curr_distance < self.game.dog_sense_radius

    @Explanation('WHAT')
    def seeing_other_agent(self, prev_obs, curr_obs):
        """
        This WHAT explanation returns True if another agent is in the agent's view.
        """
        neighbours = self.separate_neighbours_by_type(curr_obs["neighbours"])
        return Dog in neighbours

    @Explanation('WHAT')
    def seeing_nothing(self, prev_obs, curr_obs):
        """
        This WHAT explanation returns True if the agent is isolated.
        """
        neighbours = self.separate_neighbours_by_type(curr_obs["neighbours"])
        return Dog not in neighbours and Sheep not in neighbours

    @Explanation('WHERE')
    def left_side(self, prev_obs, curr_obs):
        """
        This WHERE explanation returns True if the agent is close to the left edge of the map.
        """
        return curr_obs["agent_pos"][0] < self.game.dog_sense_radius

    @Explanation('WHERE')
    def right_side(self, prev_obs, curr_obs):
        """
        This WHERE explanation returns True if the agent is close to the right edge of the map.
        """
        return self.game.map_side - curr_obs["agent_pos"][0] < self.game.dog_sense_radius

    @Explanation('WHERE')
    def bottom_side(self, prev_obs, curr_obs):
        """
        This WHERE explanation returns True if the agent is close to the bottom edge of the map.
        """
        return self.game.map_side - curr_obs["agent_pos"][1] < self.game.dog_sense_radius

    @Explanation('WHERE')
    def top_side(self, prev_obs, curr_obs):
        """
        This WHERE explanation returns True if the agent is close to the top edge of the map.
        """
        return curr_obs["agent_pos"][1] < self.game.dog_sense_radius

    @Explanation('WHERE')
    def centre(self, prev_obs, curr_obs):
        """
        This WHERE explanation returns True if the agent is close to the centre of the map.
        """
        centre_point = (self.game.map_side / 2, self.game.map_side / 2)
        distance_to_centre = distance.euclidean(curr_obs["agent_pos"], centre_point)
        return distance_to_centre < self.game.dog_sense_radius

    @Explanation('WHERE')
    def red_zone(self, prev_obs, curr_obs):
        """
        This WHERE explanation returns True if the agent is outside the pasture zone.
        """
        centre_point = (self.game.map_side / 2, self.game.map_side / 2)
        distance_to_centre = distance.euclidean(curr_obs["agent_pos"], centre_point)
        return distance_to_centre > self.game.map_side / 2.0

    @Explanation('WHERE')
    def elsewhere(self, prev_obs, curr_obs):
        """
        This WHERE explanation returns True if the agent is neither at the centre or at the edges of the map.
        """
        p, c = prev_obs, curr_obs
        return not (self.left_side(p, c)   or
                    self.right_side(p, c)  or
                    self.top_side(p, c)    or
                    self.bottom_side(p, c) or
                    self.centre(p, c)      or
                    self.red_zone(p, c))



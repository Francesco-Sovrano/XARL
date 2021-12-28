import numpy as np
import math
import pygame
import random
from scipy.spatial import KDTree, distance

from particle import Particle


class Sheep(Particle):
    color = (122, 122, 122)  # Grey

    def __init__(self, x, y, radius=5.0):
        super().__init__(x, y, radius)

    def move(self, vector):
        self.x = self.pos[0] + vector[0]
        self.y = self.pos[1] + vector[1]

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)


class Dog(Particle):
    color = (235, 165, 27)  # Yellow

    def __init__(self, x, y, radius=5.0):
        super().__init__(x, y, radius)

    def move(self, vector):
        self.x = self.pos[0] + vector[0]
        self.y = self.pos[1] + vector[1]

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)


class Pen(Particle):
    color = (0, 162, 232)  # Light blue

    def __init__(self, x, y, radius=5.0):
        super().__init__(x, y, radius)

    def move(self, vector):
        pass

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)


class ShepherdGame:
    def __init__(self, num_dogs, num_sheep, render=True, save_frames=False):
        self.num_dogs = num_dogs
        self.num_sheep = num_sheep
        self.sheep_size = 6.0
        self.dog_size = 5.0
        self.pen_radius = self.sheep_size * 10.0
        self.dt = 0.15
        self.sheep_sense_radius = 75.0
        self.dog_sense_radius = 150.0
        self.render = render
        self.save_frames = save_frames

        self.dogs = []
        self.sheep = []
        self.sheep_pen = None
        self.pasture = None

        self.total_reward = 0

        self.particles = []

        # KD-tree with particles to be updated at each frame.
        self.particle_tree = None

        self.generate_map()

        self.frame_count = 0

        if self.render:
            self.init_render()

    def add_to_game(self, particle):
        if type(particle) == Sheep:
            self.sheep.append(particle)
        elif type(particle) == Dog:
            self.dogs.append(particle)
        elif type(particle) == Pen:
            if self.sheep_pen is not None:
                self.particles.remove(self.sheep_pen)
            self.sheep_pen = particle

        self.particles.append(particle)

    def generate_map(self, map_sparsity=30):

        self.map_side = map_sparsity * self.num_sheep
        if self.map_side < 400:
            self.map_side = 400
        self.map_centre = (self.map_side / 2.0, self.map_side / 2.0)

        random_angle = random.uniform(0, 2*np.pi)
        random_distance = random.uniform(0, (self.map_side / 2) - self.pen_radius)
        pen_x = self.map_centre[0] + random_distance * math.cos(random_angle)
        pen_y = self.map_centre[1] + random_distance * math.sin(random_angle)

        sheep_pen = Pen(x=pen_x, y=pen_y, radius=self.pen_radius)
        self.add_to_game(sheep_pen)

        placed_sheep = 0
        placed_dogs = 0

        def valid_place(particle):
            for dog in self.dogs:
                if particle.overlaps(dog):
                    return False
            for sheep in self.sheep:
                if particle.overlaps(sheep):
                    return False
            if particle.overlaps(self.sheep_pen):
                return False
            # No overlaps established.
            distance_to_centre = distance.euclidean(particle.pos, self.map_centre)
            if distance_to_centre > self.map_side / 2.0:
                return False

            return True

        while placed_sheep < self.num_sheep:
            sheep = Sheep(random.uniform(0, self.map_side), random.uniform(0, self.map_side), self.sheep_size)
            if valid_place(sheep):
                placed_sheep += 1
                self.add_to_game(sheep)

        while placed_dogs < self.num_dogs:
            dog = Dog(random.uniform(0, self.map_side), random.uniform(0, self.map_side), self.dog_size)
            if valid_place(dog):
                placed_dogs += 1
                self.add_to_game(dog)

        self.update_KD_tree()

    def init_render(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.map_side, self.map_side),
                                              pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
        self.screen.fill((122, 22, 22))  # Fill with red.

    def draw_screen(self):
        # self.screen.fill((22, 120, 52))  # Fill with green.

        self.screen.fill((122, 22, 22))  # Fill with red.
        pygame.draw.circle(self.screen, color=(22, 120, 52), center=self.map_centre, radius=self.map_side / 2.0)

        # Draw sheep and dogs.
        for sheep in self.sheep:
            sheep.draw(self.screen)
        for dog in self.dogs:
            dog.draw(self.screen)

        # Draw sheep pen (represented as a blue circle).
        self.sheep_pen.draw(self.screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        pygame.display.update()


    def count_sheep(self, count_and_remove=True):
        saved = 0
        lost = 0
        to_remove = []
        inside_pen = self.get_neighbours_of(self.sheep_pen)
        for particle in inside_pen:
            # Don't delete dogs or pens.
            if type(particle) == Sheep:
                saved += 1
                to_remove.append(particle)
        for particle in self.sheep:
            distance_to_centre = distance.euclidean(particle.pos, self.map_centre)
            if distance_to_centre > self.map_side / 2.0:
                lost += 1
                to_remove.append(particle)
        if count_and_remove:
            for particle in to_remove:
                self.sheep.remove(particle)
                self.particles.remove(particle)
        return saved, lost

    def update_KD_tree(self):
        points = [(p.x, p.y) for p in self.particles]
        self.particle_tree = KDTree(points)

    def get_neighbours_of(self, particle, radius=None):
        if not radius:
            radius = particle.radius
        indices = self.particle_tree.query_ball_point(particle.pos, radius)
        particles = []
        for index in indices:
            if index < len(self.particles):
                particles.append(self.particles[index])

        # Sort by distance.
        particles.sort(key=lambda x: distance.euclidean(particle.pos, x.pos))
        return particles

    def step(self, action_dict):
        states, rewards, explanation = {}, {}, {}

        self.count_sheep()

        for agent_id, action in action_dict.items():
            self.local_act(agent_id, action)

        for sheep in self.sheep:
            self.move_sheep(sheep)

        for sheep in self.sheep:
            sheep.advance(self.dt)

        self.update_KD_tree()

        # self.total_reward += reward

        if self.render:
            self.draw_screen()
            if self.save_frames:
                self.frame_count += 1
                pygame.image.save(self.screen, "frames/screen{:04d}.png".format(self.frame_count))

    def move_sheep(self, sheep):
        neighbours = self.get_neighbours_of(sheep, self.sheep_sense_radius)
        sheep.velocity = np.zeros(2, dtype=np.float32)

        def bound(value, low, high):
            if low > high:
                low, high = high, low
            return max(low, min(high, value))

        def calculate_field_velocity(sheep, particle, max_distance):
            v = np.zeros(2, dtype=np.float32)
            px, py = particle.x, particle.y
            sx, sy = sheep.x, sheep.y
            d = distance.euclidean((px, py), (sx, sy))
            df = bound(d / max_distance, 0.0, 1.0)
            if type(particle) == Sheep:
                # Small attractive force.
                desired_heading = math.atan2(py - sy, px - sx)
                # Repulsive force to avoid collisions.
                intensity = 0.2 if d > 3 * sheep.radius else -0.2
                if d < 2 * sheep.radius:
                    intensity = -30.0
            elif type(particle) == Dog:
                # Strong repulsive force.
                desired_heading = math.atan2(sy - py, sx - px)
                intensity = 3.0
            else:
                return np.zeros(2, dtype=np.float32)

            v = np.array([df * np.cos(desired_heading), df * np.sin(desired_heading)]) * intensity
            return v

        for neighbour in neighbours:
            sheep.velocity += calculate_field_velocity(sheep, neighbour, max_distance=self.sheep_sense_radius)

    def local_act(self, agent_id, action):
        action = np.array((1.0, 1.0))  # Fixed motion for test
        self.dogs[agent_id].velocity = action
        self.dogs[agent_id].advance(self.dt)

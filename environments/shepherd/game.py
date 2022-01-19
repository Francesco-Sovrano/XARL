import numpy as np
import math
import random
import os
from scipy.spatial import KDTree, distance

from environments.shepherd.particle import Particle


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
        self.sheep_size = 1.0
        self.dog_size = 1.0
        self.pen_radius = self.sheep_size * 10.0
        self.dt = 0.4
        self.sheep_sense_radius = 15.0
        self.dog_sense_radius = 30.0
        self.render = render
        self.save_frames = save_frames
        self.base_grid = None
        self.global_grid = None

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
            # Only load pygame if we're actually rendering things.
            globals()["pygame"] = __import__("pygame")
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

    def generate_map(self, map_sparsity=5):

        self.map_side = map_sparsity * self.num_sheep

        self.map_centre = (self.map_side / 2.0, self.map_side / 2.0)

        # Add sheep pen to random part of green space.
        random_angle = random.uniform(0, 2 * np.pi)
        random_distance = random.uniform(0, (self.map_side / 2) - self.pen_radius)
        pen_x = self.map_centre[0] + random_distance * math.cos(random_angle)
        pen_y = self.map_centre[1] + random_distance * math.sin(random_angle)

        sheep_pen = Pen(x=pen_x, y=pen_y, radius=self.pen_radius)
        self.add_to_game(sheep_pen)

        # Generate grid map
        dim = self.map_side
        self.base_grid = np.zeros(shape=(dim, dim), dtype=np.int8)
        radius = self.map_side / 2
        for x in range(self.map_side):
            for y in range(self.map_side):
                if distance.euclidean((x,y), self.map_centre) > radius:
                    # Adding red cells.
                    self.base_grid[x][y] = 1
                if distance.euclidean((x,y), self.sheep_pen.pos) < self.pen_radius:
                    # Adding blue cells (pen).
                    self.base_grid[x][y] = 2

        self.global_grid = np.copy(self.base_grid)

        # Place moving particles.
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

        # Draw sheep pen (represented as a blue circle).
        self.sheep_pen.draw(self.screen)

        # Draw sheep and dogs.
        for sheep in self.sheep:
            sheep.draw(self.screen)
        for dog in self.dogs:
            dog.draw(self.screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        pygame.display.update()
        # delay = 1
        # print(f"Frame {self.frame_count} drawn. Waiting {delay} seconds. Sheep count {len(self.sheep)}.")
        # pygame.time.delay(delay*1000)



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

    def compute_global_grid(self):
        """
        Converts an obs dict into an agent_view grid.
        Cell values:
        0: empty pasture cell (green)
        1: empty external cell (red)
        2: pen cell (blue)
        3: sheep
        4: other dogs
        5: agent (self)
        """
        self.global_grid = np.copy(self.base_grid)
        coord_space = np.linspace(0.5, self.map_side-0.5, self.map_side-1, dtype=np.float32)
        for particle in self.particles:
            x = np.searchsorted(coord_space, particle.x)
            y = np.searchsorted(coord_space, particle.y)
            if type(particle) == Sheep:
                self.global_grid[x][y] = 3
            elif type(particle) == Dog:
                self.global_grid[x][y] = 4

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
        self.compute_global_grid()

        # self.total_reward += reward
        self.frame_count += 1
        if self.render:
            self.draw_screen()
            if self.save_frames:
                if not os.path.exists('frames'):
                    os.makedirs('frames')
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
                desired_heading = math.atan2(py - sy, px - sx)
                # Small attractive force to encourage flocking.
                # Repulsive force to avoid collisions.
                intensity = 0.2 if d > 3 * sheep.radius else -0.2
                if d < 2 * sheep.radius:
                    intensity = -3.0
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

        # Avoid agent going out of bounds
        x = self.dogs[agent_id].x
        y = self.dogs[agent_id].y
        if x < 0.0:
            x = 0.0
        elif x > self.map_side:
            x = self.map_side
            
        if y < 0.0:
            y = 0.0
        elif y > self.map_side:
            y = self.map_side
        self.dogs[agent_id].pos[0] = x
        self.dogs[agent_id].pos[1] = y
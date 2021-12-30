from itertools import count
import numpy as np
from scipy.spatial import KDTree, distance


class Particle:
    """A class representing a two-dimensional particle."""
    _ids = count(0)
    MAX_MOVE = 0.0
    def __init__(self, x, y, radius=0.01):
        """Initialize the particle's position, velocity, and radius.
        """

        # Particle id.
        self.pid = next(self._ids)
        self.pos = np.array((x, y), dtype=np.float32)
        self.velocity = np.array((0.0, 0.0), dtype=np.float32)
        self.radius = radius
        self.mass = self.radius**2
        self.solid = True

    def __hash__(self):
        return hash(self.pid)


    # For convenience, map the components of the particle's position and
    # velocity vector onto the attributes x, y, vx and vy.
    @property
    def x(self):
        return self.pos[0]
    @x.setter
    def x(self, value):
        self.pos[0] = value
    @property
    def y(self):
        return self.pos[1]
    @y.setter
    def y(self, value):
        self.pos[1] = value
    @property
    def vx(self):
        return self.velocity[0]
    @vx.setter
    def vx(self, value):
        self.velocity[0] = value
    @property
    def vy(self):
        return self.velocity[1]
    @vy.setter
    def vy(self, value):
        self.velocity[1] = value

    def overlaps(self, other):
        """Does the circle of this Particle overlap that of other?"""

        return np.hypot(*(self.pos - other.pos)) < self.radius + other.radius

    def contains(self, other):
        """Does the circle of this Particle fully contains that of other?"""
        return self.radius >= other.radius and np.hypot(*(self.pos - other.pos)) + other.radius <= self.radius

    def advance(self, dt):
        """Advance the Particle's position forward in time by dt."""
        old_pos = np.copy(self.pos)
        self.pos += self.velocity * dt
        d = distance.euclidean((old_pos[0], old_pos[1]), (self.pos[0], self.pos[1]))
        if d > self.MAX_MOVE:
            self.MAX_MOVE = d
            print("New maximum move: ", d)


import numpy as np
from src.env.constants import *
from src.physics.forces import wall_repulsion
from src.utils.math import limit_magnitude, normalize

class Sheep:
    def __init__(self, position):
        self.pos = np.array(position, dtype=np.float32)
        angle = np.random.uniform(0, 2 * np.pi)
        self.vel = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

    def update(self, sheep_list, dog_pos):
        separation = np.zeros(2)
        alignment = np.zeros(2)
        cohesion = np.zeros(2)
        count = 0

        for other in sheep_list:
            if other is self:
                continue
            diff = self.pos - other.pos
            dist = np.linalg.norm(diff)

            if dist < SEPARATION_RADIUS and dist > 0:
                separation += normalize(diff) / dist

            if dist < NEIGHBOR_RADIUS:
                alignment += other.vel
                cohesion += other.pos
                count += 1

        if count > 0:
            alignment = normalize(alignment / count)
            cohesion = normalize((cohesion / count) - self.pos)

        # Dog repulsion
        dog_vec = self.pos - dog_pos
        dog_dist = np.linalg.norm(dog_vec)
        dog_force = np.zeros(2)
        if dog_dist < DOG_REPULSION_RADIUS and dog_dist > 0:
            dog_force = normalize(dog_vec) * (DOG_REPULSION_RADIUS / dog_dist)

        noise = np.random.uniform(-1, 1, size=2)

        wall_force = wall_repulsion(self.pos)

        acceleration = (
            SEPARATION_WEIGHT * separation +
            ALIGNMENT_WEIGHT * alignment +
            COHESION_WEIGHT * cohesion +
            DOG_REPULSION_WEIGHT * dog_force +
            wall_force +
            NOISE_WEIGHT * noise
        )

        self.vel += acceleration
        self.vel = limit_magnitude(self.vel, SHEEP_MAX_SPEED)
        self.pos += self.vel
        

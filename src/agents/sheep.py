import numpy as np
from src.env.constants import *
from src.physics.forces import wall_repulsion
from src.utils.math import normalize, limit_magnitude

class Sheep:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=np.float32)
        angle = np.random.uniform(0, 2 * np.pi)
        self.vel = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

    def update(self, sheep, dogs):
        separation = np.zeros(2)
        alignment = np.zeros(2)
        cohesion = np.zeros(2)
        count = 0

        for other in sheep:
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

        dog_force = np.zeros(2)
        for dog_pos, bark in dogs:
            diff = self.pos - dog_pos
            dist = np.linalg.norm(diff)
            if dist < DOG_REPULSION_RADIUS and dist > 0:
                strength = (DOG_REPULSION_RADIUS / dist) * (1 + bark * BARK_MULTIPLIER)
                dog_force += normalize(diff) * strength

        accel = (
            SEPARATION_WEIGHT * separation +
            ALIGNMENT_WEIGHT * alignment +
            COHESION_WEIGHT * cohesion +
            DOG_REPULSION_WEIGHT * dog_force +
            wall_repulsion(self.pos) +
            NOISE_WEIGHT * np.random.uniform(-1, 1, 2)
        )

        self.vel += accel
        self.vel = limit_magnitude(self.vel, SHEEP_MAX_SPEED)
        self.pos += self.vel

        self.pos[0] = np.clip(self.pos[0], 0, FIELD_WIDTH)
        self.pos[1] = np.clip(self.pos[1], 0, FIELD_HEIGHT)

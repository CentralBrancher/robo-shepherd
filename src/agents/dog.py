import numpy as np
from src.env.constants import *
from src.agents.base import KinematicBody

class Dog(KinematicBody):
    def __init__(self, pos, bark_cooldown_max=30):
        super().__init__(pos, DOG_MAX_SPEED)
        self.accel_scale = DOG_ACCEL
        self.bark = 0.0
        self.bark_cooldown = 0
        self.bark_cooldown_max = bark_cooldown_max

    def step(self, action):
        move = np.clip(action[:2], -1, 1)
        requested_bark = np.clip(action[2], 0, 1)

        if self.bark_cooldown == 0 and requested_bark > 0:
            self.bark = requested_bark
            self.bark_cooldown = self.bark_cooldown_max
        else:
            self.bark = 0.0

        if self.bark_cooldown > 0:
            self.bark_cooldown -= 1

        accel = move * self.accel_scale
        self.integrate(accel)

        # Hard wall handling (no sticking)
        for i, limit in enumerate([FIELD_WIDTH, FIELD_HEIGHT]):
            if self.pos[i] <= 0 or self.pos[i] >= limit:
                self.vel[i] = 0
                self.pos[i] = np.clip(self.pos[i], 0, limit)

        return self.bark

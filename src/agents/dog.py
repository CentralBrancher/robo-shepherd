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

        self.pos[0] = np.clip(self.pos[0], 0, FIELD_WIDTH)
        self.pos[1] = np.clip(self.pos[1], 0, FIELD_HEIGHT)

        return self.bark

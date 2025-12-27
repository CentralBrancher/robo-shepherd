import numpy as np
from src.env.constants import *
from src.agents.base import KinematicBody

class Dog(KinematicBody):
    def __init__(self, pos):
        super().__init__(pos, DOG_MAX_SPEED)
        self.accel_scale = DOG_ACCEL

    def step(self, action):
        """
        action = [dx, dy, bark]
        """
        move = np.clip(action[:2], -1, 1)
        bark = np.clip(action[2], 0, 1)

        accel = move * self.accel_scale
        self.integrate(accel)

        self.pos[0] = np.clip(self.pos[0], 0, FIELD_WIDTH)
        self.pos[1] = np.clip(self.pos[1], 0, FIELD_HEIGHT)

        return bark

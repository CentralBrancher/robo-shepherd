import numpy as np
from src.env.constants import *
from src.utils.math import limit_magnitude

class Dog:
    def __init__(self, position=None):
        if position is None:
            position = (FIELD_WIDTH / 2, FIELD_HEIGHT / 2)

        self.pos = np.array(position, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.accel = DOG_ACCEL

    # -------------------------
    # Core physics (shared)
    # -------------------------
    def step(self, action):
        """
        action: np.array([ax, ay]) in [-1, 1]
        """
        action = np.clip(action, -1.0, 1.0)

        self.vel += action * self.accel
        self.vel = limit_magnitude(self.vel, DOG_MAX_SPEED)
        self.pos += self.vel

        # Hard wall bounds (safety net)
        self.pos[0] = np.clip(self.pos[0], 0, FIELD_WIDTH)
        self.pos[1] = np.clip(self.pos[1], 0, FIELD_HEIGHT)

    # -------------------------
    # Manual control wrapper
    # -------------------------
    def update(self, keys):
        import pygame  # local import (important!)

        action = np.zeros(2, dtype=np.float32)

        if keys[pygame.K_w]:
            action[1] -= 1
        if keys[pygame.K_s]:
            action[1] += 1
        if keys[pygame.K_a]:
            action[0] -= 1
        if keys[pygame.K_d]:
            action[0] += 1

        if np.linalg.norm(action) > 0:
            action /= np.linalg.norm(action)

        self.step(action)

    # -------------------------
    # RL entry point
    # -------------------------
    def apply_action(self, action):
        self.step(action)

    # -------------------------
    # Reset
    # -------------------------
    def reset(self, position=None):
        if position is None:
            position = (FIELD_WIDTH / 2, FIELD_HEIGHT / 2)

        self.pos = np.array(position, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)

import numpy as np
import pygame
from src.env.constants import *
from src.utils.math import limit_magnitude

class Dog:
    def __init__(self, position):
        self.pos = np.array(position, dtype=np.float32)
        self.vel = np.zeros(2)

    def update(self, keys):
        move = np.zeros(2)

        if keys[pygame.K_w]:
            move[1] -= 1
        if keys[pygame.K_s]:
            move[1] += 1
        if keys[pygame.K_a]:
            move[0] -= 1
        if keys[pygame.K_d]:
            move[0] += 1

        if np.linalg.norm(move) > 0:
            move = move / np.linalg.norm(move)

        self.vel = move * DOG_MAX_SPEED
        self.pos += self.vel

        self.pos[0] = np.clip(self.pos[0], 0, FIELD_WIDTH)
        self.pos[1] = np.clip(self.pos[1], 0, FIELD_HEIGHT)

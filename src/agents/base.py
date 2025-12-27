import numpy as np
from src.utils.math import limit_magnitude

class KinematicBody:
    def __init__(self, pos, max_speed):
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.zeros_like(self.pos)
        self.max_speed = max_speed

    def integrate(self, accel):
        self.vel += accel
        self.vel *= 0.95        
        self.vel = limit_magnitude(self.vel, self.max_speed)
        self.pos += self.vel

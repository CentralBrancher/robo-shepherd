import numpy as np
from src.env.constants import *

class Gate:
    def __init__(self):
        self.reset()

    def reset(self):
        self.center = np.array([
            np.random.uniform(FIELD_WIDTH * 0.7, FIELD_WIDTH - 40),
            np.random.uniform(100, FIELD_HEIGHT - 100)
        ], dtype=np.float32)
        self.width = np.random.uniform(60, 120)

    def contains(self, pos):
        return (
            abs(pos[0] - self.center[0]) < self.width / 2
            and abs(pos[1] - self.center[1]) < 10
        )

class World:
    def __init__(self):
        self.gate = Gate()

    def reset(self):
        self.gate.reset()

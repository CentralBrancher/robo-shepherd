import numpy as np
from src.env.constants import FIELD_WIDTH, FIELD_HEIGHT

class Gate:
    def __init__(self):
        self.reset()

    def reset(self):
        self.center = np.array([
            np.random.uniform(200, FIELD_WIDTH - 200),
            np.random.uniform(200, FIELD_HEIGHT - 200)
        ], dtype=np.float32)
        self.width = np.random.uniform(60, 120) 

    def contains(self, pos):
        return (
            abs(pos[0] - self.center[0]) < self.width / 2 and
            abs(pos[1] - self.center[1]) < 10
        )


class World:
    def __init__(self):
        self.gate = Gate()

    def reset(self):
        self.gate.reset()
        return self.gate.center

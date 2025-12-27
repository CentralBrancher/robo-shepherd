import numpy as np
from src.env.constants import FIELD_WIDTH, FIELD_HEIGHT

class Gate:
    def __init__(self, center, width=60):
        self.center = np.array(center, dtype=np.float32)
        self.width = width

    def contains(self, point):
        return (
            abs(point[0] - self.center[0]) < self.width / 2
            and abs(point[1] - self.center[1]) < 10
        )


class World:
    def __init__(self):
        self.width = FIELD_WIDTH
        self.height = FIELD_HEIGHT

        # Gate on right edge
        self.gate = Gate(
            center=(FIELD_WIDTH - 30, FIELD_HEIGHT / 2),
            width=80
        )

    def clamp(self, pos):
        pos[0] = np.clip(pos[0], 0, self.width)
        pos[1] = np.clip(pos[1], 0, self.height)
        return pos

    def reset(self):
        # Gate is static for now, nothing to reset
        pass


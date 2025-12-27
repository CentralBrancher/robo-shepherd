from src.env.constants import *
from src.env.world import World
from src.physics.flocking import sheep_in_gate

class EpisodeState:
    def __init__(self, sheep_count):
        self.steps = 0
        self.success_counter = 0
        self.sheep_count = sheep_count
        self.done = False
        self.success = False

    def update(self, sheep, world):
        self.steps += 1

        in_gate = sheep_in_gate(sheep, world.gate)
        ratio = in_gate / self.sheep_count

        if ratio >= SUCCESS_SHEEP_RATIO:
            self.success_counter += 1
        else:
            self.success_counter = 0

        if self.success_counter >= SUCCESS_FRAMES:
            self.done = True
            self.success = True

        if self.steps >= MAX_EPISODE_STEPS:
            self.done = True

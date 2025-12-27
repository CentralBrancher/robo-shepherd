import pygame
import pygame.font
import numpy as np

from src.env.constants import *
from src.physics.flocking import compute_centroid, compute_flock_radius

GREEN = (30, 120, 30)
WHITE = (230, 230, 230)
RED = (200, 50, 50)
BLUE = (50, 150, 255)
YELLOW = (255, 200, 0)

class PygameRenderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((FIELD_WIDTH, FIELD_HEIGHT))
        pygame.display.set_caption("Robo Shepherd")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

    def render(self, sheep, dog, world):
        self.clock.tick(FPS)
        self.screen.fill(GREEN)

        # Gate
        gate = world.gate
        rect = pygame.Rect(
            int(gate.center[0] - gate.width // 2),
            int(gate.center[1] - 10),
            int(gate.width),
            20
        )

        pygame.draw.rect(
            self.screen,
            BLUE,
            rect,
            2
        )

        # Sheep
        for s in sheep:
            pygame.draw.circle(self.screen, WHITE, s.pos.astype(int), SHEEP_RADIUS)

        # Dog
        pygame.draw.circle(self.screen, RED, dog.pos.astype(int), DOG_RADIUS)

        # Debug: centroid + radius
        centroid = compute_centroid(sheep)
        radius = compute_flock_radius(sheep, centroid)

        pygame.draw.circle(self.screen, YELLOW, centroid.astype(int), 4)
        pygame.draw.circle(self.screen, YELLOW, centroid.astype(int), int(radius), 1)

        # HUD
        text = f"Sheep in gate: {sum(world.gate.contains(s.pos) for s in sheep)} / {len(sheep)}"
        surface = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(surface, (10, 10))

        pygame.display.flip()

    def close(self):
        pygame.quit()



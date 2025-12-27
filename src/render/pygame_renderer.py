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

    def render(self, sheep, dogs, world):
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

        pygame.draw.rect(self.screen, BLUE, rect, 2)

        # Sheep
        for s in sheep:
            pygame.draw.circle(self.screen, WHITE, s.pos.astype(int), SHEEP_RADIUS)

        # Dog
        for d in dogs:
            if d.bark > 0:  # actively barking
                color = (
                    int(RED[0] + (YELLOW[0] - RED[0]) * d.bark),
                    int(RED[1] + (YELLOW[1] - RED[1]) * d.bark),
                    int(RED[2] + (YELLOW[2] - RED[2]) * d.bark)
                )
            elif d.bark_cooldown > 0:  # cooling down
                cooldown_ratio = d.bark_cooldown / d.bark_cooldown_max
                # fade from RED (ready) to ORANGE (cooling)
                color = (
                    int(RED[0] + (255 - RED[0]) * cooldown_ratio),
                    int(RED[1] + (165 - RED[1]) * cooldown_ratio),
                    int(RED[2] + (0 - RED[2]) * cooldown_ratio)
                )
            else:
                color = RED  # idle

            pygame.draw.circle(self.screen, color, d.pos.astype(int), DOG_RADIUS)

            # Optional halo for active bark
            if d.bark > 0:
                halo_radius = DOG_RADIUS + int(d.bark * 10)
                pygame.draw.circle(self.screen, YELLOW, d.pos.astype(int), halo_radius, 2)

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



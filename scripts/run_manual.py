import pygame
import numpy as np

from src.env.constants import *
from src.agents.sheep import Sheep
from src.agents.dog import Dog

pygame.init()
screen = pygame.display.set_mode((FIELD_WIDTH, FIELD_HEIGHT))
pygame.display.set_caption("Robo Shepherd – MVP")
clock = pygame.time.Clock()

# Create agents
sheep = [
    Sheep((np.random.uniform(0, FIELD_WIDTH),
           np.random.uniform(0, FIELD_HEIGHT)))
    for _ in range(NUM_SHEEP)
]

dog = Dog((FIELD_WIDTH / 2, FIELD_HEIGHT / 2))

running = True
while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    dog.update(keys)

    for s in sheep:
        s.update(sheep, dog.pos)

    # Render
    screen.fill((30, 120, 30))

    # Sheep
    for s in sheep:
        pygame.draw.circle(
            screen, (230, 230, 230),
            s.pos.astype(int),
            SHEEP_RADIUS
        )

    # Dog
    pygame.draw.circle(
        screen, (200, 50, 50),
        dog.pos.astype(int),
        DOG_RADIUS
    )

    pygame.display.flip()

pygame.quit()

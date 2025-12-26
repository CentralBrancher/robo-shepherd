import pygame
import numpy as np

from src.env.constants import *
from src.agents.sheep import Sheep
from src.agents.dog import Dog
from src.env.world import World
from src.render.pygame_renderer import PygameRenderer
from src.rewards.shaping import compute_reward
from src.env.shepherd_env import EpisodeState

renderer = PygameRenderer()
world = World()

sheep = [
    Sheep((np.random.uniform(0, FIELD_WIDTH),
           np.random.uniform(0, FIELD_HEIGHT)))
    for _ in range(NUM_SHEEP)
]

episode = EpisodeState(len(sheep))

dog = Dog((FIELD_WIDTH / 2, FIELD_HEIGHT / 2))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    dog.update(keys)

    for s in sheep:
        s.update(sheep, dog.pos)

    reward = compute_reward(sheep, world)
    episode.update(sheep, world)

    renderer.render(sheep, dog, world)

    if episode.done:
        print("SUCCESS!" if episode.success else "FAILED")
        pygame.time.wait(2000)
        running = False

pygame.quit()

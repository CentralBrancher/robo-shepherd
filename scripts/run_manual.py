import pygame
import numpy as np
from src.env.gym_shepherd_env import ShepherdGymEnv

pygame.init()
env = ShepherdGymEnv(render_mode="human")
obs, _ = env.reset()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    action = np.zeros(env.action_space.shape)

    action[0] = keys[pygame.K_d] - keys[pygame.K_a]
    action[1] = keys[pygame.K_s] - keys[pygame.K_w]
    action[2] = keys[pygame.K_SPACE]  # bark

    obs, _, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()

env.close()
pygame.quit()

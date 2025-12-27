import numpy as np
import pygame
from src.env.gym_shepherd_env import ShepherdGymEnv

def main():
    env = ShepherdGymEnv(
        sheep_count=10,
        render_mode="human"
    )

    obs, info = env.reset()
    done = False
    clock = pygame.time.Clock()

    print("Starting random policy episode...")

    while not done:
        # Sample random continuous action
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        clock.tick(60)  # lock FPS for sanity

    print("Episode finished")
    print("Success:", info.get("success", False))

    env.close()


if __name__ == "__main__":
    main()

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.env.world import World
from src.env.constants import *
from src.env.shepherd_env import EpisodeState
from src.agents.dog import Dog
from src.agents.sheep import Sheep
from src.rewards.shaping import compute_reward


class ShepherdGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, sheep_count=10, render_mode=None):
        super().__init__()

        self.sheep_count = sheep_count
        self.render_mode = render_mode

        # World
        self.world = World()
        self.dog = Dog((FIELD_WIDTH / 2, FIELD_HEIGHT / 2))
        self.sheep = [
            Sheep((
                np.random.uniform(0, FIELD_WIDTH),
                np.random.uniform(0, FIELD_HEIGHT)
            ))
            for _ in range(self.sheep_count)
        ]
        self.episode = EpisodeState(sheep_count)

        # Action: continuous dog movement
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Observation size
        obs_size = (
            2 +  # dog pos
            2 +  # dog vel
            sheep_count * 2 +  # sheep pos
            sheep_count * 2 +  # sheep vel
            2  # gate pos
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.renderer = None
        if render_mode == "human":
            from src.render.pygame_renderer import PygameRenderer
            self.renderer = PygameRenderer()

    # -------------------------
    # Gym API
    # -------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.world.reset()
        self.dog.reset()
        self.sheep = [
            Sheep((
                np.random.uniform(0, FIELD_WIDTH),
                np.random.uniform(0, FIELD_HEIGHT)
            ))
            for _ in range(self.sheep_count)
        ]
        self.episode = EpisodeState(self.sheep_count)

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        # Clip for safety
        action = np.clip(action, -1.0, 1.0)

        # Apply action
        self.dog.apply_action(action)

        # Update sheep
        for s in self.sheep:
            s.update(self.sheep, self.dog.pos)

        # Reward
        reward = compute_reward(self.sheep, self.world)

        # Episode logic
        self.episode.update(self.sheep, self.world)

        terminated = self.episode.success
        truncated = self.episode.done and not self.episode.success

        obs = self._get_obs()
        info = {"success": self.episode.success}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.renderer:
            self.renderer.render(self.sheep, self.dog, self.world)

    def close(self):
        if self.renderer:
            self.renderer.close()

    # -------------------------
    # Helpers
    # -------------------------

    def _get_obs(self):
        obs = []

        # Dog
        obs.extend(self.dog.pos)
        obs.extend(self.dog.vel)

        # Sheep
        for s in self.sheep:
            obs.extend(s.pos)
        for s in self.sheep:
            obs.extend(s.vel)

        # Gate
        obs.extend(self.world.gate.center)

        return np.array(obs, dtype=np.float32)

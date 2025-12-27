import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.env.constants import *
from src.env.world import World
from src.env.shepherd_env import EpisodeState
from src.agents.dog import Dog
from src.agents.sheep import Sheep
from src.rewards.shaping import compute_reward

class ShepherdGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.world = World()
        self.dogs = [Dog((FIELD_WIDTH / 2, FIELD_HEIGHT / 2))]
        self.sheep = [Sheep((np.random.rand()*FIELD_WIDTH, np.random.rand()*FIELD_HEIGHT))
                      for _ in range(NUM_SHEEP)]
        self.episode = EpisodeState(NUM_SHEEP)

        self.action_space = spaces.Box(-1, 1, (NUM_DOGS * 3,), np.float32)

        obs_size = (
            NUM_DOGS * 4 +
            NUM_SHEEP * 4 +
            2
        )

        self.observation_space = spaces.Box(-1, 1, (obs_size,), np.float32)

        self.renderer = None
        if render_mode == "human":
            from src.render.pygame_renderer import PygameRenderer
            self.renderer = PygameRenderer()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        for d in self.dogs:
            d.pos[:] = FIELD_WIDTH/2, FIELD_HEIGHT/2
            d.vel[:] = 0
        self.sheep = [Sheep((np.random.rand()*FIELD_WIDTH, np.random.rand()*FIELD_HEIGHT))
                      for _ in range(NUM_SHEEP)]
        self.episode = EpisodeState(NUM_SHEEP)
        return self._obs(), {}

    def step(self, action):
        dogs_data = []
        for i, dog in enumerate(self.dogs):
            bark = dog.step(action[i*3:(i+1)*3])
            dogs_data.append((dog.pos.copy(), bark))

        for s in self.sheep:
            s.update(self.sheep, dogs_data)

        reward = compute_reward(self.sheep, self.world)
        self.episode.update(self.sheep, self.world)

        if self.renderer:
            self.renderer.render(self.sheep, self.dogs, self.world)

        return self._obs(), reward, self.episode.success, self.episode.done, {"success": self.episode.success}

    def _obs(self):
        obs = []
        for d in self.dogs:
            obs.extend(d.pos)
            obs.extend(d.vel)
        for s in self.sheep:
            obs.extend(s.pos)
            obs.extend(s.vel)
        obs.extend(self.world.gate.center)
        return np.array(obs, np.float32)

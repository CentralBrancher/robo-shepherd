import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.env.constants import *
from src.env.world import World
from src.env.shepherd_env import EpisodeState
from src.agents.dog import Dog
from src.agents.sheep import Sheep
from src.rewards.shaping import compute_reward
from src.physics.flocking import compute_centroid

class ShepherdGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.world = World()
        self.dogs = [Dog((np.random.rand()*FIELD_WIDTH, np.random.rand()*FIELD_HEIGHT))
                        for _ in range(NUM_DOGS)]
        self.sheep = [Sheep((np.random.rand()*FIELD_WIDTH, np.random.rand()*FIELD_HEIGHT))
                        for _ in range(NUM_SHEEP)]
        self.episode = EpisodeState(NUM_SHEEP)

        self.action_space = spaces.Box(-1, 1, (NUM_DOGS * 3,), np.float32)
        obs_size = NUM_DOGS * 9  # dog pos/vel + centroid vector + gate vector + flock radius
        self.observation_space = spaces.Box(-1, 1, (obs_size,), np.float32)

        self.renderer = None
        if render_mode == "human":
            from src.render.pygame_renderer import PygameRenderer
            self.renderer = PygameRenderer()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        self.dogs = [Dog((np.random.rand()*FIELD_WIDTH, np.random.rand()*FIELD_HEIGHT))
                      for _ in range(NUM_DOGS)]
        self.sheep = [Sheep((np.random.rand()*FIELD_WIDTH, np.random.rand()*FIELD_HEIGHT))
                      for _ in range(NUM_SHEEP)]
        self.episode = EpisodeState(NUM_SHEEP)
        self.prev_centroid = compute_centroid(self.sheep)
        return self._obs(), {}
    
    def render(self): 
        if not self.renderer: 
            return 
        
        import pygame 
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT: 
                self.close() 
                raise SystemExit 
                
        self.renderer.render(self.sheep, self.dogs, self.world)

    def step(self, action):
        dogs_data = []
        for i, dog in enumerate(self.dogs):
            bark = dog.step(action[i*3:(i+1)*3])
            dogs_data.append((dog.pos.copy(), bark))

        for s in self.sheep:
            s.update(self.sheep, dogs_data, self.world.gate.center)

        reward = compute_reward(self.sheep, self.dogs, self.world, prev_centroid=self.prev_centroid)
        if self.episode.success:
            reward += 500.0

        self.prev_centroid = compute_centroid(self.sheep)

        self.episode.update(self.sheep, self.world)

        if self.render_mode == "human":
            self.render()

        info = {
            "success": self.episode.success,
            "sheep_in_gate": sum(self.world.gate.contains(s.pos) for s in self.sheep),
            "reward": reward
        }

        return self._obs(), reward, self.episode.success, self.episode.done, info

    def _obs(self):
        obs = []
        centroid = compute_centroid(self.sheep)
        flock_radius = max(np.linalg.norm(s.pos - centroid) for s in self.sheep)
        gate_vec = self.world.gate.center - centroid

        for d in self.dogs:
            obs.extend(d.pos)
            obs.extend(d.vel)
            obs.extend(centroid - d.pos)
            obs.extend(gate_vec)
            obs.append(flock_radius)

        return np.array(obs, np.float32)

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from tqdm import tqdm
import numpy as np

class ProgressCallback(BaseCallback):
    """Displays progress bar with avg reward, avg steps, and sheep in gate."""
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.pbar = None
        self.episode_steps = None
        self.episode_rewards = None
        self.episode_sheep = None

    def _on_training_start(self):
        n_envs = self.training_env.num_envs
        self.episode_steps = np.zeros(n_envs, dtype=int)
        self.episode_rewards = np.zeros(n_envs, dtype=float)
        self.episode_sheep = np.zeros(n_envs, dtype=float)
        total_steps = n_envs * self.locals.get('n_steps', 0)
        self.pbar = tqdm(total=total_steps, desc="Training", ncols=120)

    def _on_step(self):
        infos = self.locals.get("infos", [])

        for i, info in enumerate(infos):
            self.episode_steps[i] += 1
            self.episode_rewards[i] += info.get("reward", 0)
            self.episode_sheep[i] = info.get("sheep_in_gate", self.episode_sheep[i])
            if info.get("success", False) or info.get("done", False):
                self.episode_steps[i] = 0
                self.episode_rewards[i] = 0
                self.episode_sheep[i] = 0

        avg_reward = np.mean([r for r in self.episode_rewards if r > 0]) if np.any(self.episode_rewards > 0) else 0
        avg_steps = np.mean([s for s in self.episode_steps if s > 0]) if np.any(self.episode_steps > 0) else 0
        avg_sheep = np.mean([s for s in self.episode_sheep if s > 0]) if np.any(self.episode_sheep > 0) else 0

        self.pbar.set_postfix({
            "AvgReward": f"{avg_reward:.2f}",
            "AvgSteps": f"{avg_steps:.0f}",
            "SheepInGate": f"{avg_sheep:.1f}"
        })

        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()


def make_checkpoint_callback(save_freq=25_000, save_path="./models/", name_prefix="ppo_shepherd"):
    """Return a Stable-Baselines3 CheckpointCallback."""
    return CheckpointCallback(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix)

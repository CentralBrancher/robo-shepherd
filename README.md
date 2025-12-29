# Robo Shepherd (RL Herding Environment)

---

Robo Shepherd is a **reinforcement-learning research environment** where autonomous dogs learn to herd a flock of sheep into a gate using **continuous control**, **multi-agent dynamics**, and **shaped rewards**.

The project is designed to study:

* Long-horizon control
* Collective motion and flocking
* Reward shaping and exploit avoidance
* PPO stability in physics-driven environments

At its current stage, the environment is **fully trainable**, **visually interactive**, and supports **resumable PPO training**.

---

## Core Objective

Train one or more dogs to:

1. Gather a dispersed flock of sheep
2. Drive the flock centroid toward a randomly placed gate
3. Maintain herd cohesion
4. Avoid exploiting walls or pinning behavior
5. Sustain success for a fixed number of frames

---

## Learning Setup

* **Algorithm:** PPO (Stable-Baselines3)
* **Action space:** Continuous (movement + bark per dog)
* **Observation space:**

  * Dog position & velocity
  * Flock centroid vector
  * Gate direction
  * Flock radius
* **Episodes:** Up to 3000 steps or early success termination

Training supports:

* Parallel environments (VecEnv)
* Checkpointing
* Safe resume after interruption

---

## Environment Dynamics

### Sheep

* Boids-style flocking:

  * Separation
  * Alignment
  * Cohesion
* Repelled by nearby dogs
* Bark increases repulsion strength
* Subject to soft noise

### Dogs

* Kinematic point agents
* Continuous acceleration
* Speed-limited
* Bark with cooldown
* No steering at walls (hard constraints)

### World

* Fixed-size field
* Randomized gate position & width
* Success requires ≥80% of sheep in gate for 60 frames

---

## Reward Design (Current)

The reward function is explicitly shaped to prevent known RL exploits.

Main components:

* **Centroid progress toward gate** (primary driver)
* **Time penalty** (encourages efficiency)
* **Flock compactness pressure**
* **Gate blocking penalty** (dogs in front of flock)
* **Anti wall-hugging penalties** (dogs & sheep)
* **Centroid stagnation penalty**
* **Terminal success bonus** (applied externally)

The reward actively discourages:

* Wall pinning
* Compress-and-wait strategies
* Sideways herding
* Gate blocking

---

## Rendering & Interaction

* Pygame-based visual renderer
* Displays:

  * Sheep
  * Dogs (with bark state visualization)
  * Gate
  * Flock centroid & radius
  * HUD with sheep-in-gate count

Supports:

* Manual keyboard control
* Random policy rollout
* PPO evaluation playback

---

## Project Structure

```text
robo-shepherd/
├── models/                # PPO checkpoints
├── scripts/
│   ├── train_ppo.py       # Training / resume script
│   ├── test_ppo.py        # Evaluation
│   ├── run_manual.py      # Keyboard control
│   ├── run_random.py      # Random baseline
│   └── callbacks.py       # Progress + checkpoints
├── src/
│   ├── agents/            # Dog & sheep logic
│   ├── env/               # Gymnasium environment
│   ├── physics/           # Flocking & pressure models
│   ├── rewards/           # Reward shaping
│   ├── render/            # Pygame renderer
│   └── utils/             # Math helpers
└── README.md
```

---

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train PPO (from scratch or resume)

```bash
python -m scripts.train_ppo
```

Supports:

* Automatic checkpointing
* Resume via `PPO.load()`

### 3. Manual control

```bash
python -m scripts.run_manual
```

Controls:

* `WASD`: Move
* `Space`: Bark

### 4. Evaluate trained model

```bash
python -m scripts.test_ppo
```

---

## Current Status

* ✔ Environment stable
* ✔ Reward exploits mitigated
* ✔ PPO training resumable
* ✔ Visual debugging tools
* ✔ Multi-dog support

**Ongoing tuning:**

* Reward coefficient balancing
* Bark strategy refinement
* Role specialization (driver / flanker)

---

## Research Directions (Planned)

* Curriculum learning (no walls → walls)
* Soft wall forces instead of clipping
* Learned dog roles
* Population-based training
* Self-play variants
* Emergent communication via bark


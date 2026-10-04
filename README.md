# 🏆 America's Cup Multi-Agent Reinforcement Learning Simulator

An advanced two-dimensional sailing simulator inspired by the **America's Cup**, where two autonomous boats learn continuous rudder and sail control to compete in a tactical match race.

The system is based on **Multi-Agent Reinforcement Learning (MARL)** and is implemented as a parallel PettingZoo environment, trained with the **PPO (Proximal Policy Optimization)** algorithm using Stable-Baselines3 and SuperSuit.

---

## 📖 Program Overview

The simulator recreates a close match race between two identical boats, named `red_boat` and `blue_boat`. To win, the agents must navigate a stochastic racecourse while respecting sailing physics, the boats' velocity polar diagrams, and real-world right-of-way rules.

### ⚓ Race Structure (Legs)

The race is organized into sequential stages controlled by the agents' internal state (`current_leg`):

1. **Stage 0: Pre-start & Line Crossing**: Boats spawn below the starting gate ($Y \approx 120$m). They must line up and cross the **Bottom Gate** ($Y = 200$m) between the marks. Crossing too early or missing the gate results in immediate disqualification.
2. **Leg 1: Upwind**: Boats sail north toward the **Top Gate** ($Y = 3900$m). Since a sailboat cannot sail directly into the wind (the no-go zone), agents must learn to tack while maximizing *Velocity Made Good (VMG)*.
3. **Leg 1.5: Mark Rounding**: After passing the top gate, agents must round one of the outer marks (port or starboard) to turn the bow south.
4. **Leg 2: Downwind**: Agents sail with the wind toward the finish at the **Bottom Gate** ($Y = 200$m). The race ends as soon as the first boat successfully crosses the final gate.

### ⛵ Simulator Physics and Environmental Dynamics

The simulation core implements models derived from high-performance racing boats:

* **Velocity Polar Diagrams (VPP)**: The theoretical maximum speed depends on the true wind angle (TWA) and wind strength. The simulator dynamically calculates polar diagrams for two sailing modes: **Displacement** (the boat is in the water, slower but able to sail closer to the wind) and **Foiling** (the boat flies on its hydrofoils, much faster but with a wider no-go zone).
* **Foiling Mechanics**: The boat takes off onto its foils when its speed exceeds **18 knots** and returns to displacement mode if speed falls below **15 knots**. Transitions include temporary penalties and account for state inertia ($I_F = 0.98$, $I_D = 0.85$).
* **Sail Trim**: Sail aerodynamic efficiency is modeled by a Gaussian curve centered on the optimal trim for the current point of sail. Agents control sail trim continuously.
* **Spatiotemporal Wind Field**: Wind is not constant. A base wind varies over time through a *random walk* (from 15 to 22 knots), alongside a perturbed $10 \times 10$ spatial grid that models the creation, evolution, and decay of local gusts and wind shifts using stochastic *mean-reverting* processes.
* **Collisions & Right-of-Way Rules (Rule 10)**: Boats have a physical collision radius of 20 meters and a safety zone of 40 meters. The simulator calculates predictive penalties based on *Time-To-Collision (TTC)* and heavily penalizes violations of right-of-way on opposite tacks. The port-tack boat (sailing with the wind coming from its left) receives a 1.6× penalty multiplier or immediate disqualification in the event of a severe collision.

---

## 📂 Project Structure

```text
Multi-agent_America_Cup/
├── core/                   # Physics model and environmental dynamics
│   ├── boat_physics.py     # Polar speed, VMG, and kinematic updates
│   ├── sail_trim.py        # Sail-trim optimization and efficiency calculation
│   └── wind_model.py       # Spatial grid and stochastic wind random walk
├── env/                    # Gymnasium/PettingZoo-style environment
│   ├── sailing_env.py      # State management, leg logic, rewards, and collisions
│   └── rendering.py        # Environment visualization
├── report/                 # Academic report in LaTeX
│   ├── report.tex          # Report source with detailed theory
│   └── report.pdf          # Compiled report
├── images/                 # Charts and visual assets for the report and README
├── videos/                 # Output directory for rendered MP4 simulations
├── config.yaml             # Simulation parameters and RL hyperparameters
├── main.py                 # Main CLI for training, testing, and video generation
├── train_ppo.py            # Parallelized PPO training routine
├── evaluate_ppo.py         # Policy evaluation and trajectory rendering
├── callbacks.py            # Custom metrics logging and checkpointing
└── requirements.txt        # Project dependencies
```

---

## 🚀 How to Use

### 🛠️ Environment Setup

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # .venv\Scripts\activate   # Windows
   ```
2. Install the dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *Note: To save videos as MP4 files, make sure FFmpeg is installed on your operating system.*

### 🧠 Model Training

Training uses 16 configurable parallel processes to collect samples efficiently and relies on PettingZoo's *self-play* mechanism. The CLI automatically saves successive model versions in `models/`.

```bash
# Start a new training run from scratch (overwrites previous temporary checkpoints)
python main.py --train-new

# Resume training from the latest saved model
python main.py --train-resume

# Start training with custom command-line parameters
python main.py --train-new --steps 2000000 --n-envs 16 --model-name sailing_model
```

### 📺 Evaluation and Video Rendering

Evaluate a trained policy by racing it in the simulator and recording the episode to an MP4 file:

```bash
# Run one test episode and export it as a demo video
python main.py --video-file videos/sailing_demo.mp4

# Run a test suite of five races with different stochastic seeds
python main.py --test-multi
```

### 📊 TensorBoard Monitoring

The `SuccessTrackingCallback` class in `callbacks.py` continuously records custom performance metrics, including leg completion rate, upwind speed, average sail-trim efficiency, average collision penalties, and counts for each type of failure or disqualification.

To inspect training progress:

```bash
tensorboard --logdir ./sailing_tensorboard/
```

---

## 🔬 Configuration Details

Physical and algorithmic parameters are managed centrally in [config.yaml](config.yaml). The main hyperparameters are:

| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| `learning_rate` | $2\times 10^{-4}$ | Learning rate for the Adam optimizer |
| `n_steps` | $4096$ | Steps per environment before each PPO update |
| `batch_size` | $1024$ | Minibatch size for gradient updates |
| `net_arch` | `[256, 256]` | Two-layer MLP for the policy ($\pi$) and value function ($V$) |
| `frame_stack` | $4$ | Number of stacked frames to capture changes in wind and opponent state |
| `success_threshold_pct` | $0.95$ | Minimum completion rate required to trigger early stopping |
| `success_window_size` | $100$ | Number of episodes used to calculate the success rate |

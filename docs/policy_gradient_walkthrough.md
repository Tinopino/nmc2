# Policy-Gradient Single Intersection Walkthrough

The repository already includes a REINFORCE-style policy-gradient agent
(`src/sumo_rl/agents/PolicyGradient.py`) and a runnable experiment for a
single intersection (`src/experiments/PolicyGradient_single_intersection.py`).
This guide explains how the pieces fit together so you can train the agent to
minimize total waiting time.

## 1. Environment setup

1. Install SUMO and set the `SUMO_HOME` environment variable so the experiment
   can locate the SUMO Python tools.
2. Install Python dependencies (e.g., `pip install -r requirements.txt` if
   available) and ensure `numpy`, `pandas`, and `traci` are present.
3. From the repository root, run commands in an environment where `SUMO_HOME`
   and your Python interpreter are available.

## 2. How the policy-gradient agent works

* The agent parameters live in `src/sumo_rl/agents/PolicyGradient.py`.
* Observations are flattened feature vectors (`obs_dim`) and actions come from
  the discrete SUMO action space (`action_space.n`).
* `act(obs)` computes softmax probabilities over actions and samples one.
* `learn(reward, done)` stores the transition, and on episode end performs a
  classical reward-to-go REINFORCE update. Returns are discounted, centered
  around a running baseline `R_bar` estimated from raw episode returns, then
  optionally normalized before applying gradient ascent with clipping. The
  reward comes directly from the environment; in the single-intersection setup
  we explicitly use the `diff-waiting-time` reward so positive returns reflect
  decreases in accumulated waiting time.

## 3. Running the single-intersection experiment

The experiment script configures the SUMO environment and loops over episodes:

```bash
python src/experiments/PolicyGradient_single_intersection.py \
  -s 10000 \            # simulation seconds per episode
  -a 0.001 \            # policy-gradient learning rate
  -g 0.99 \              # discount factor used in reward-to-go updates
  -gc 5.0 \              # gradient clipping threshold
  -runs 3 \              # number of training episodes
  -gui                   # optional: visualize the simulation
```

Key behaviors inside the script:

* Creates an output folder under `outputs/single-intersection` and writes one
  CSV per episode with per-step metrics.
* Initializes `SumoEnvironment` with the single-intersection network and route
  files stored under `src/sumo_rl/nets/single-intersection/`.
* Builds one `PolicyGradientAgent` per traffic signal and reuses it across
  episodes so learning accumulates.
* At each step: encodes observations (`env.encode`), samples actions, steps the
  environment, then calls `agent.learn` with the observed reward. When an
  episode ends, the agent performs a discounted-return update so you should see
  average per-episode reward prints trend upward as learning progresses.
* After all episodes, combines episode CSVs into a single
  `combined_episodes_continuous.csv` for easier plotting.

## 4. Evaluating waiting-time reduction

* The environment computes waiting-time-related metrics. With `add_system_info`
  enabled (True by default), the output CSVs include `system_total_waiting_time`
  and `system_mean_waiting_time` columns representing aggregate performance.
* Monitor these columns across episodes (or plot the combined CSV) to verify
  whether the policy improves at reducing waiting time.

## 5. Tips for tuning

* Increase `-a/--alpha` (learning rate) or `beta_rew` (baseline update rate) in
  `PolicyGradientAgent` to adjust learning speed and variance reduction.
* If gradients explode or learning becomes unstable, lower `-gc/--grad_clip`
  (or raise it if updates look too small). The default 5.0 keeps updates
  bounded.
* Adjust `min_green`/`max_green` to ensure the action space aligns with realistic
  traffic light timings.
* Longer runs (`-runs` and `-s`) provide more experience but take more compute.
* If exploration stalls, experiment with a slightly larger `epsilon` or a
  decaying schedule via `-decay`.

With these steps, you can train and iterate on the gradient policy agent to
minimize total waiting time in the SUMO single-intersection scenario.

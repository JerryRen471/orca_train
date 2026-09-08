# ORCA cube reinforcement learning

This package provides two independent training paths for `orca_sim`'s cube task:

- `orca-train-drqv2`: the existing visual red-face-direction task.
- `orca-train-state`: a full-quaternion, state-based cube-reorientation baseline.

The visual command and its checkpoints retain their original observation and
checkpoint formats.

## Visual observation and action

- observation: three stacked `84 x 84` RGB frames plus 17 measured joint angles
- action: 17 normalized joint increments; `1.0` is `+3°` and `-1.0` is `-3°`
- reward and termination: passed through unchanged from
  `OrcaHandRightCubeOrientation` in `orca_sim/task_envs.py`
- augmentation: DrQ random shifts with four-pixel replicated padding

## State baseline

The state agent uses TD3 updates: a frozen target actor generates bootstrap
actions, two critic updates precede each actor update, and both target networks
track their online networks only after an actor update. Target-policy smoothing
uses fixed noise (standard deviation 0.2, clip 0.5), independently of the behavior
exploration schedule. The actor optimizes deterministic Q1. Network width,
learning rate, replay, n-step returns and the environment control period retain
the state baseline settings.

The default policy update also penalizes deviation from the actions sampled from
replay. Its loss is `-alpha * mean(Q1) / max(mean(abs(Q1)), 1) + MSE(policy, replay)`;
the Q denominator is detached from gradients. This TD3+BC-inspired constraint
limits exploitation of actions with little replay support. The denominator floor
avoids amplifying a nearly zero early critic. `--behavior-regularization-alpha`
defaults to 0.1; `--no-behavior-regularization` recovers the unregularized TD3
objective for comparisons. This online adaptation is evaluated separately from
the original offline TD3+BC algorithm.

Checkpoints identify the regularized algorithm as `state_td3_bc_v1` and save both
target networks and the critic-update counter. Unregularized runs use
`state_td3_v1`. Full resume requires the same algorithm and regularization alpha;
a dimension-compatible actor can still be transferred with `--warm-start`, which
synchronizes its target actor and starts fresh critics. Earlier 64-value state
checkpoints support actor transfer but cannot restore their old critic/optimizers
into the new algorithm.

The state trainer uses a bounded 64-value observation in this order:

1. 17 actuator positions normalized by joint ROM.
2. 17 actuator velocities scaled by a configured limit.
3. Cube position in the fixed hand-mount frame.
4. Cube linear velocity in the hand-mount frame.
5. Cube angular velocity in the hand-mount frame.
6. The canonical relative target quaternion in MuJoCo `wxyz` order.
7. The 17 position-controller targets normalized by joint ROM.

Actions accumulate onto the previous controller targets: zero action preserves
the load-bearing setpoint instead of following joint sag. The wrist setpoint is
held by default (16 policy actions); `--no-fix-wrist` enables all 17 joints. State
checkpoints record `observation_mode`, state dimension, and action dimension;
loading a visual checkpoint into the state trainer, or vice versa, is rejected
with a compatibility error. Old 47-value state checkpoints are incompatible with
this controller and must not be resumed or used to initialize the new policy.

The static curricula are:

| Preset | Target policy | Sequence | Tolerance | Target duration | Hold | Control period |
|---|---|---:|---:|---:|---:|---:|
| `turn_30` | fixed +30° hand-X turn | 1 | 15° | 8.0 s | 2.0 s | 0.08 s |
| `turn_45` | fixed +45° hand-X turn | 1 | 15° | 8.0 s | 2.0 s | 0.08 s |
| `turn_60` | fixed +60° hand-X turn | 1 | 15° | 8.0 s | 2.0 s | 0.08 s |
| `single_goal` | fixed +90° hand-X turn | 1 | 0.4 rad | 8.0 s | 2.0 s | 0.08 s |
| `right_angle` | random ±90° hand axis | 1 | 0.4 rad | 8.0 s | 2.0 s | 0.08 s |
| `multi_goal` | shuffled 24-orientation bag | 20 | 15° | 8.0 s | 2.0 s | 0.08 s |

Each preset uses angular-error progress reward. Training does not advance between
presets automatically; start a new run explicitly after evaluating the previous
stage. Reset holds the initial controller targets for 1 second before drawing a
goal from the settled cube orientation. The initial XY position is independently
jittered by up to 1 mm per axis, using the reset seed. The fixed turns and the
orientation bag use this settled reference; random turns use the current pose.
`--reset-settle-duration-s`, `--cube-pos-xy-jitter`, and
`--no-target-relative-to-reset` allow controlled comparisons.

## Install

```bash
cd /Users/jerry/Code/orca_train
uv sync --extra dev
```

MuJoCo off-screen rendering on macOS uses the system graphics session, so run the
trainer from a normal logged-in Terminal rather than an SSH-only session.

## Train the visual baseline

```bash
cd /Users/jerry/Code/orca_train
uv run orca-train-drqv2 \
  --device mps \
  --fix-wrist \
  --total-steps 1000000 \
  --output-dir runs/cube_flip_seed1
```

Use `--device cuda` on an NVIDIA training machine or `--device cpu` for debugging.
The default replay window is 20,000 transitions and uses about 1.3 GB for images.
Increase `--replay-capacity` only if sufficient RAM is available.

Training writes:

- `config.json`: the complete run configuration
- `metrics.jsonl`: episode return, success rate, and optimization losses
- `checkpoint_<step>.pt`: network and optimizer state

Resume a run with:

```bash
uv run orca-train-drqv2 \
  --device mps \
  --total-steps 1000000 \
  --resume runs/cube_flip_seed1/checkpoint_50000.pt \
  --output-dir runs/cube_flip_seed1
```

## Train the state baseline

Start with the 30° curriculum and advance through `turn_45`, `turn_60`,
`single_goal`, `right_angle`, then `multi_goal` after each stage reaches a
stable evaluation success rate:

```bash
cd /Users/jerry/Code/orca_train
uv run orca-train-state \
  --preset turn_30 \
  --device mps \
  --total-steps 1000000 \
  --eval-episodes 100 \
  --output-dir runs/state_cube_turn_30_support_seed1
```

Use `--device cuda` on an NVIDIA machine or `--device cpu` for a short
debugging run. The resolved preset and every explicit override are written to
`config.json`. Without `--output-dir`, runs are separated by preset and seed as
`runs/state_cube_<preset>_seed<seed>`. Evaluation reports target counts, seconds
per completed target, final/best angular error, drop/timeout rates, a 95% Wilson
interval for success rate, and
90°/120°/180° target buckets. It also records the 60°/45°/30°/success-tolerance
funnel, drop step, maximum stable-success streak, and the cube height and speed
when the tolerance is first reached. Evaluation defaults to 100 fixed-seed
episodes. It reports the number of distinct initial state vectors and suppresses
the confidence interval if any initial states repeat. Each run also evaluates a
zero-action controller on the same seeds and logs success-rate gain over that
baseline. A course must outperform zero action and meet the full hold requirement
before promotion; a high angle-only funnel rate is insufficient.

Evaluation also reports action RMS, the fraction of action components above 0.95
in magnitude, and the fraction of active joint targets/positions within 1% of
either ROM limit. The fixed wrist is excluded. `mean_value_bias_terminal`
compares the initial twin-critic minimum with the realized discounted return,
using only episodes that terminate. Externally truncated episodes are excluded
because their future return is unobserved; `value_calibration_episodes` records
the comparison count. These diagnostics measure value error and aggressive
control directly rather than inferring them from success rate alone.

The default exploration policy uses `linear(0.2,0.05,100000)` Gaussian noise
clipped to `0.2`; the replay warm-up samples only within `±0.1`. Angular progress
is rewarded only with sufficient height, hand contact and bounded linear/angular
speed. Negative angular progress remains penalized when grasp is lost. The
`0.01` height shaping term is supplemented by a `0.1` reward on each stable step
inside target tolerance. These values can be overridden for controlled ablations.

Promote a **new-format** checkpoint with actor-only transfer:

```bash
uv run orca-train-state --preset turn_45 --device cuda \
  --warm-start runs/state_cube_turn_30_support_seed1/checkpoint_100000.pt \
  --total-steps 100000 --output-dir runs/state_cube_turn_45_support_seed1
```

`--warm-start` resets critics, optimizers, replay, step count, and exploration for
the new course. `--resume` restores the actor, critics, optimizers and saved step
count for the same course; its `--total-steps` is the cumulative stopping point.
Replay is not checkpointed, so resume refills replay for `--seed-steps` before
learning, while preserving the saved noise-schedule progress. Both modes require
the same observation/action dimensions and network width. Use a fresh output
directory for every experiment or resumed segment to keep logs comparable.

A four-step smoke run verifies environment construction, replay insertion,
gradient updates, evaluation, checkpoint save, and checkpoint load. It does not
measure learned manipulation performance.

## Verify

```bash
uv run pytest
```

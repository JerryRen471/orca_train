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

The state trainer uses a bounded 47-value observation in this order:

1. 17 actuator positions normalized by joint ROM.
2. 17 actuator velocities scaled by a configured limit.
3. Cube position in the fixed hand-mount frame.
4. Cube linear velocity in the hand-mount frame.
5. Cube angular velocity in the hand-mount frame.
6. The canonical relative target quaternion in MuJoCo `wxyz` order.

Actions remain normalized measured-state-relative joint increments. State
checkpoints record `observation_mode`, state dimension, and action dimension;
loading a visual checkpoint into the state trainer, or vice versa, is rejected
with a compatibility error.

The static curricula are:

| Preset | Target policy | Sequence | Tolerance | Target duration | Hold | Control period |
|---|---|---:|---:|---:|---:|---:|
| `single_goal` | fixed +90° hand-X turn | 1 | 0.4 rad | 8.0 s | 0.16 s | 0.08 s |
| `right_angle` | random ±90° hand axis | 1 | 0.4 rad | 8.0 s | 0.16 s | 0.08 s |
| `multi_goal` | shuffled 24-orientation bag | 20 | 15° | 8.0 s | 0.16 s | 0.08 s |

Each preset uses angular-error progress reward. Training does not advance between
presets automatically; start a new run explicitly after evaluating the previous
stage.

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

Start with the deterministic single-goal curriculum:

```bash
cd /Users/jerry/Code/orca_train
uv run orca-train-state \
  --preset single_goal \
  --device mps \
  --total-steps 1000000 \
  --output-dir runs/state_cube_single_goal_seed1
```

Use `--device cuda` on an NVIDIA machine or `--device cpu` for a short
debugging run. The resolved preset and every explicit override are written to
`config.json`. Without `--output-dir`, runs are separated by preset and seed as
`runs/state_cube_<preset>_seed<seed>`. Evaluation reports target counts, seconds
per completed target, final/best angular error, drop/timeout rates, and
90°/120°/180° target buckets.

A four-step smoke run verifies environment construction, replay insertion,
gradient updates, evaluation, checkpoint save, and checkpoint load. It does not
measure learned manipulation performance.

## Verify

```bash
uv run pytest
```

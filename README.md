# ORCA visual reinforcement learning

This package trains a DrQ-v2 agent on `orca_sim`'s cube-orientation task. The task
starts with the cube's red face down and succeeds when the red face points up.

## Observation and action

- observation: three stacked `84 x 84` RGB frames plus 17 measured joint angles
- action: 17 normalized joint increments; `1.0` is `+3°` and `-1.0` is `-3°`
- reward and termination: passed through unchanged from
  `OrcaHandRightCubeOrientation` in `orca_sim/task_envs.py`
- augmentation: DrQ random shifts with four-pixel replicated padding

## Install

```bash
cd /Users/jerry/Code/orca_train
uv sync --extra dev
```

MuJoCo off-screen rendering on macOS uses the system graphics session, so run the
trainer from a normal logged-in Terminal rather than an SSH-only session.

## Train

```bash
cd /Users/jerry/Code/orca_train
uv run orca-train-drqv2 \
  --device mps \
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

## Verify

```bash
uv run pytest
```

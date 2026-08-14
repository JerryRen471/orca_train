# Original Orca Sim Baseline Design

## Objective

Train a clean comparison baseline using the repository version of `orca_sim` without the experimental reward parameters or normalized proprioception added in later experiments.

## Environment

- Use `OrcaHandRightCubeOrientation` exactly as stored at the current `orca_sim` repository `HEAD`.
- Preserve the original absolute alignment reward, lift bonus, drop penalty of `1.0`, success condition, drop condition, and 200-step horizon.
- Load this source from an isolated Spark directory so current local and server experiments remain intact.
- Retain the camera correction already committed in the selected repository revision because RGB observations require a usable camera.

## Training Adapter

- Observation: 84×84 RGB, three-frame stack, plus 17 raw joint angles in radians.
- Action: 17 normalized policy outputs in `[-1, 1]`, mapped to joint target increments of at most 3 degrees and clipped to actuator ranges.
- Wrist remains trainable.
- Reset starts with the red face down and keeps the existing 0.01 m cube XY jitter used by the visual training adapter.

## Training

- Algorithm: existing DrQ-v2 implementation in `orca_train`.
- Start from scratch with seed 1 for 1,000,000 environment steps.
- Evaluate for five episodes every 10,000 steps.
- Save a checkpoint every 50,000 steps.
- Use CUDA and MuJoCo EGL on Spark.

## Isolation and Verification

- Do not overwrite current `orca_sim` or experiment runs.
- Create a separate source snapshot and output directory on Spark.
- Before the full run, verify the loaded task constructor has no experimental reward arguments, proprioception contains raw radians, the action dimension is 17, and a 300-step CUDA smoke run completes.


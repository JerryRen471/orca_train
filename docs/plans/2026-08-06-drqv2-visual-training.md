# ORCA DrQ-v2 Visual Training Pipeline Implementation Plan

**Goal:** Train a DrQ-v2 policy for the ORCA v2 cube-orientation task from 84x84 RGB frame stacks plus 17 joint angles, using bounded incremental joint actions.

**Architecture:** A Gymnasium wrapper converts the existing state task into a dictionary observation while preserving its ground-truth reward. A self-contained PyTorch DrQ-v2 agent, replay buffer, and CLI trainer live in `orca_train`; `orca_sim` remains unchanged.

**Tech Stack:** Python 3.11, Gymnasium, MuJoCo, NumPy, PyTorch, pytest, uv.

## Task 1: Package and visual environment wrapper

- Create `pyproject.toml` and package structure.
- Write failing tests for observation shapes, frame stacking, reward passthrough, and ±3 degree incremental actions.
- Implement `OrcaVisualCubeEnv` and verify tests pass.

## Task 2: DrQ-v2 learning components

- Write failing tests for random-shift augmentation, replay sampling, bounded actor output, critic output, and one update step.
- Implement networks, augmentation, replay buffer, and DrQ-v2 agent.
- Verify CPU unit tests pass.

## Task 3: End-to-end training pipeline

- Write failing tests for configuration validation, evaluation, checkpoint round-trip, and a short training run with a fake environment.
- Implement CLI configuration, training/evaluation loops, JSONL metrics, checkpointing, seeding, and resume support.
- Add user documentation and example commands.

## Verification

- Run the complete `orca_train` test suite.
- Run a real MuJoCo environment smoke test without training hardware.
- Run a short CPU DrQ-v2 update against collected ORCA transitions.
- Review files and ensure no changes were made to `orca_sim`.

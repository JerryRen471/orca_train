# Original Orca Sim Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch a one-million-step DrQ-v2 baseline against an isolated, unmodified repository version of the Orca Sim cube task.

**Architecture:** Export the repository `HEAD` version of `orca_sim` into a dedicated Spark source directory, and use a baseline-specific training adapter configuration that exposes raw joint radians. Validate the exact imported class and observation/action contract before launching a new tmux training run.

**Tech Stack:** Python 3.11, Gymnasium, MuJoCo EGL, PyTorch CUDA, DrQ-v2, rsync/SSH/tmux.

## Global Constraints

- Do not overwrite current local or Spark experiment sources or run directories.
- Use raw 17-dimensional joint angles in radians, not ROM-normalized proprioception.
- Use the original repository reward and termination logic.
- Train the free-wrist 17-dimensional policy from scratch for 1,000,000 steps.

---

### Task 1: Produce and verify the isolated baseline source

**Files:**
- Source: `/Users/jerry/Code/orca_sim/src/orca_sim/task_envs.py` at Git `HEAD`
- Create on Spark: `/home/jerry/Code/orca_sim_original_baseline/src/orca_sim/`
- Create on Spark: `/home/jerry/Code/orca_train_baseline/src/orca_train/env.py`

**Interfaces:**
- Consumes: repository `HEAD` archive and existing `OrcaVisualCubeEnv` delta-action behavior.
- Produces: an import path containing the original `OrcaHandRightCubeOrientation` and raw-radian visual adapter.

- [ ] Export the committed Orca Sim source without local experimental changes.
- [ ] Create an isolated visual adapter whose `_proprio()` returns the 17 raw actuator joint angles.
- [ ] Verify with introspection that the task constructor has no `reward_mode` or configurable `drop_penalty` parameter.
- [ ] Instantiate the adapter under EGL and assert pixel shape `(9, 84, 84)`, proprio shape `(17,)`, action shape `(17,)`, and raw proprio values equal simulator qpos values.

### Task 2: Smoke test and launch the full baseline

**Files:**
- Create on Spark: `/home/jerry/Code/orca_train/runs/smoke_original_orca_sim_baseline/`
- Create on Spark: `/home/jerry/Code/orca_train/runs/cube_flip_original_orca_sim_seed1/`

**Interfaces:**
- Consumes: isolated baseline import paths from Task 1.
- Produces: checkpoints, `config.json`, `metrics.jsonl`, and `train.log` for the baseline run.

- [ ] Run 300 CUDA training steps with 50 seed steps and batch size 32.
- [ ] Assert `checkpoint_300.pt` exists and update metrics are present.
- [ ] Start a detached tmux session named `orca_drqv2_original_baseline` for 1,000,000 steps, seed 1, evaluation every 10,000 steps, and checkpoints every 50,000 steps.
- [ ] Verify the Python process, CUDA memory allocation, config file, and initial metrics output.


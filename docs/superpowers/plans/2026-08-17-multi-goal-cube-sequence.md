# Multi-Goal Cube Sequence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a goal-conditioned DrQ-v2 policy that completes 20 balanced random red-face orientation targets before an episode succeeds.

**Architecture:** ORCA Sim owns the shuffled six-direction target bag, stable target completion, reward events, counters, and terminal reasons. ORCA Train appends the three-component target to normalized joint proprioception and aggregates multi-target metrics; the agent and replay buffer continue deriving input size dynamically.

**Tech Stack:** Python 3.11, NumPy, Gymnasium, MuJoCo, PyTorch, pytest, uv.

## Global Constraints

- Targets are the six signed world axes and use shuffled bags without consecutive duplicates.
- A target requires 15-degree accuracy, cube height at least 0.12 m, linear speed at most 0.15 m/s, angular speed at most 2 rad/s, held for 10 steps.
- Sequence success requires 20 targets; each target has a 200-step limit.
- Drop below 0.10 m has priority over completion and timeout.
- Observation is three 84x84 RGB frames plus 17 normalized joints and the 3D target direction.
- Existing checkpoints are incompatible; training starts from scratch.

---

### Task 1: Goal Sequence and Terminal State in ORCA Sim

**Files:**
- Modify: `/Users/jerry/Code/orca_sim/src/orca_sim/task_envs.py`
- Test: `/Users/jerry/Code/orca_sim/tests/test_drop_penalty.py`

**Interfaces:**
- Produces: constructor parameters `target_sequence_length: int = 20` and `max_task_steps: int = 200`.
- Produces: `_draw_next_target(excluded_direction: np.ndarray | None = None) -> np.ndarray` and `_target_alignment() -> float`.
- Produces: `info` fields `target_direction`, `target_completed`, `completed_target_direction`, `completed_target_steps`, `tasks_completed`, `task_elapsed_steps`, and `termination_reason`.

- [ ] **Step 1: Write failing sequence tests**

Add tests that reset a seeded environment, draw twelve targets, assert each six-target bag equals all signed axes, assert no adjacent targets match, and assert the initial target differs from the initial red-face normal.

- [ ] **Step 2: Run sequence tests and verify RED**

Run: `PYTHONPATH=src /Users/jerry/Code/orca_train/.venv/bin/python -m pytest tests/test_drop_penalty.py -q`

Expected: FAIL because sequence constructor parameters, target bag, and generic target alignment do not exist.

- [ ] **Step 3: Implement target bag and reset state**

Add the six constant unit directions, positive integer validation, shuffled-bag drawing through `self.np_random`, initial-target exclusion, generic target alignment, and reset-time state initialization.

- [ ] **Step 4: Write failing lifecycle tests**

Cover completion at the tenth stable step, non-terminal switching for targets 1 through 19, terminal `sequence_complete` on target 20, exact timeout at step 200, and drop priority over simultaneous completion/timeout.

- [ ] **Step 5: Run lifecycle tests and verify RED**

Run the same focused pytest command and confirm failures are caused by missing multi-target lifecycle behavior.

- [ ] **Step 6: Implement cached lifecycle events**

Refactor `_update_task_status()` so it is the only counter-mutating method. Cache per-step alignment delta before switching targets; reset the new target baseline immediately after switching. Make reward, termination, truncation, observation, and info methods pure readers of cached state.

- [ ] **Step 7: Run focused ORCA Sim tests and verify GREEN**

Run: `PYTHONPATH=src /Users/jerry/Code/orca_train/.venv/bin/python -m pytest tests/test_drop_penalty.py -q`

Expected: all focused tests pass.

### Task 2: Goal-Conditioned Visual Observation

**Files:**
- Modify: `/Users/jerry/Code/orca_train/src/orca_train/env.py`
- Modify: `/Users/jerry/Code/orca_train/tests/test_env.py`

**Interfaces:**
- Consumes: underlying reset/step `info["target_direction"]` with shape `(3,)`.
- Produces: observation `proprio` with shape `(20,)`, containing normalized 17-joint values followed by the target vector.

- [ ] **Step 1: Write failing 20-value observation tests**

Update the fake cube environment to return a known target from reset and a different target from step. Assert reset and step observations contain the corresponding target in `proprio[17:]` and that the observation space is bounded to 20 values.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_env.py -q`

Expected: FAIL because proprioception still has 17 values.

- [ ] **Step 3: Implement target propagation**

Store the latest target from underlying `info`, validate shape `(3,)`, concatenate it in `_proprio()`, and update the observation space to shape `(20,)`. On a completion step, consume the already-switched target returned by ORCA Sim.

- [ ] **Step 4: Run visual environment tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_env.py -q`

Expected: all visual environment tests pass.

### Task 3: Training Configuration and Evaluation Metrics

**Files:**
- Modify: `/Users/jerry/Code/orca_train/src/orca_train/train.py`
- Modify: `/Users/jerry/Code/orca_train/tests/test_train.py`

**Interfaces:**
- Produces: `TrainConfig.target_sequence_length: int = 20` and `TrainConfig.max_task_steps: int = 200` plus matching CLI flags.
- Consumes: completion fields from Task 1.
- Produces: evaluation keys `return`, `success_rate`, `mean_tasks_completed`, `total_tasks_completed`, `drop_rate`, `timeout_rate`, and `mean_steps_per_completed_target`.

- [ ] **Step 1: Write failing evaluation and configuration tests**

Make the evaluation fake emit target completions with durations followed by sequence-complete, dropped, or timed-out endings. Assert fixed seeds and exact aggregate metrics. Assert new training defaults and CLI-to-environment propagation.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_train.py -q`

Expected: FAIL because multi-target metrics and configuration fields are absent.

- [ ] **Step 3: Implement configuration and metric aggregation**

Pass both new parameters into `OrcaVisualCubeEnv`, add `--target-sequence-length` and `--max-task-steps`, accumulate completed target duration events during evaluation, and add task count plus terminal reason to training episode records.

- [ ] **Step 4: Run training tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_train.py -q`

Expected: all training tests pass.

### Task 4: Integrated Verification

**Files:**
- Verify all files modified above.

**Interfaces:**
- Consumes: Tasks 1 through 3.
- Produces: a tested implementation ready for a separate Spark smoke run and launch decision.

- [ ] **Step 1: Run both complete test suites**

Run ORCA Sim: `PYTHONPATH=src /Users/jerry/Code/orca_train/.venv/bin/python -m pytest tests -q`

Run ORCA Train: `.venv/bin/python -m pytest tests -q`

Expected: zero failures in both repositories.

- [ ] **Step 2: Run diff and branch checks**

Run `git diff --check`, confirm both repositories are on `feature/multi-goal-cube-sequence-implementation`, and confirm no unrelated files changed.

- [ ] **Step 3: Commit each repository independently**

ORCA Sim commit: `Add multi-goal cube orientation task`

ORCA Train commit: `Add goal-conditioned sequence training`

- [ ] **Step 4: Preserve the current server experiment**

Do not stop, overwrite, or reuse the current `orca_stable_success` tmux session or `runs/cube_flip_stable_success_seed1`. Report the verified implementation and request approval before launching a new multi-goal Spark run.

# DrQ-v2 Training Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the cube-flip objective, control state, replay returns, actor update, and evaluation stability before retraining.

**Architecture:** Keep the existing environment/agent boundaries. Make the environment Markov by deriving delta targets from measured qpos, aggregate three-step transitions before replay insertion, and align actor/evaluation behavior with DrQ-v2.

**Tech Stack:** Python, Gymnasium, MuJoCo, NumPy, PyTorch, pytest.

## Global Constraints

- Preserve 84×84 RGB with three-frame stacking, 17 ROM-normalized joint angles, 17 free-wrist actions, and ±3° action increments.
- Use the approved progress/success/drop reward.
- Replay capacity is 200,000 and n-step horizon is 3.
- Evaluation uses the same 20 seeds at every checkpoint.

---

### Task 1: Environment state and reward contract

**Files:** Modify `src/orca_train/env.py`, `src/orca_sim/task_envs.py`; test `tests/test_env.py`, `tests/test_drop_penalty.py`.

- [ ] Add failing tests for measured-qpos delta actions and approved reward terms.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement the minimal environment and reward changes.
- [ ] Run focused tests and confirm they pass.

### Task 2: Three-step replay and actor correction

**Files:** Modify `src/orca_train/replay.py`, `src/orca_train/agent.py`, `src/orca_train/train.py`; test `tests/test_drqv2.py`, `tests/test_train.py`.

- [ ] Add failing tests for three-step discounted transitions and noisy twin-Q actor optimization.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement n-step aggregation, 200k capacity, target-Q metrics, and twin-Q actor loss.
- [ ] Run focused tests and confirm they pass.

### Task 3: Stable evaluation and deployment verification

**Files:** Modify `src/orca_train/train.py`; test `tests/test_train.py`.

- [ ] Add failing tests for fixed 20-seed evaluation and drop-rate logging.
- [ ] Implement and run the full local test suites.
- [ ] Sync only corrected source files to Spark and run a short CUDA smoke test.
- [ ] Start a new isolated one-million-step run only after smoke verification succeeds.


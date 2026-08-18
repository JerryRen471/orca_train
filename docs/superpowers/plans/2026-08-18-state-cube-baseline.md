# State Cube Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a full-quaternion, state-observation cube-reorientation baseline with physical-time task semantics and a separate feed-forward training command, while preserving the existing visual red-face task.

**Architecture:** `orca_sim` owns quaternion math, target sampling, task rewards, success/timeout events, and physical control timing. `orca_train` adds an independent 47-value state adapter, vector replay, a deterministic twin-Q state agent, curriculum resolution, evaluation, and a separate CLI. Existing face-direction observations, visual wrappers, commands, and checkpoint format are not changed.

**Tech Stack:** Python 3.10+, NumPy, Gymnasium, MuJoCo, PyTorch, pytest, uv.

**Spec:** [`docs/superpowers/specs/2026-08-18-state-cube-baseline-design.md`](../specs/2026-08-18-state-cube-baseline-design.md)

## Global Constraints

- Work in `/Users/jerry/Code/orca_sim` and `/Users/jerry/Code/orca_train`; do not touch the unrelated dirty files in `orca_core`.
- Create `feature/state-cube-baseline` in `orca_sim`. Continue on `feature/state-cube-baseline-design` in `orca_train` so the approved spec, plan, and implementation remain together.
- Use `uv run python` and `uv run pytest`; never install into system Python.
- Preserve `goal_mode="face_direction"`, `OrcaVisualCubeEnv`, `orca-train-drqv2`, and visual checkpoint behavior.
- Add one failing test at a time, confirm the expected failure, implement the smallest complete behavior, and rerun the focused test before each commit.
- Do not start long training, remote jobs, pushes, or pull requests.

---

## Task 1: Add Tested Quaternion and Cube-Orientation Primitives

**Files:**

- Create: `/Users/jerry/Code/orca_sim/src/orca_sim/orientation.py`
- Create: `/Users/jerry/Code/orca_sim/tests/test_orientation.py`

- [ ] **Step 1: Write failing tests for quaternion validation, sign invariance, relative composition, and known angular errors.**

Add tests that require these public functions:

```python
from orca_sim.orientation import (
    canonicalize_quaternion,
    quaternion_angular_error,
    quaternion_multiply,
    relative_quaternion,
)

def test_relative_quaternion_is_canonical_and_reconstructs_target():
    current = np.array([1.0, 0.0, 0.0, 0.0])
    target = np.array([0.0, 1.0, 0.0, 0.0])
    relative = relative_quaternion(current, -target)
    assert relative[0] >= 0.0
    np.testing.assert_allclose(
        np.abs(np.dot(quaternion_multiply(current, relative), target)), 1.0
    )

@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]), np.pi / 2),
        (np.array([0.0, 1.0, 0.0, 0.0]), np.pi),
    ],
)
def test_quaternion_angular_error_matches_known_rotations(target, expected):
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    assert quaternion_angular_error(identity, target) == pytest.approx(expected)
    assert quaternion_angular_error(identity, -target) == pytest.approx(expected)
```

Also require `(4,)` shape and nonzero norm errors.

- [ ] **Step 2: Run the focused tests and confirm import failure.**

Run from `/Users/jerry/Code/orca_sim`:

```bash
uv run pytest tests/test_orientation.py -q
```

Expected: collection fails because `orca_sim.orientation` does not exist.

- [ ] **Step 3: Implement normalized `wxyz` quaternion operations.**

Implement these exact interfaces in `orientation.py`:

```python
def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray: ...
def canonicalize_quaternion(quaternion: np.ndarray) -> np.ndarray: ...
def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray: ...
def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray: ...
def relative_quaternion(current: np.ndarray, target: np.ndarray) -> np.ndarray: ...
def quaternion_angular_error(current: np.ndarray, target: np.ndarray) -> float: ...
def axis_angle_quaternion(axis: np.ndarray, angle_rad: float) -> np.ndarray: ...
```

`relative_quaternion` must compute `conjugate(current) * target` and canonicalize the result. `quaternion_angular_error` must use `2 * acos(clip(abs(dot), 0, 1))` after normalization.

- [ ] **Step 4: Add and satisfy tests for the 24 proper cube rotations.**

Add `cube_orientation_quaternions() -> np.ndarray` returning shape `(24, 4)`. Generate the 24 determinant-`+1` signed permutation matrices, convert each matrix to a normalized canonical quaternion, and sort deterministically by rounded components. Test shape, unit norm, pairwise uniqueness up to sign, and rotation-matrix determinant.

- [ ] **Step 5: Run focused tests and commit.**

```bash
uv run pytest tests/test_orientation.py -q
git diff --check
git add src/orca_sim/orientation.py tests/test_orientation.py
git commit -m "Add cube orientation quaternion utilities"
```

---

## Task 2: Add Effective Physical Control Timing

**Files:**

- Modify: `/Users/jerry/Code/orca_sim/src/orca_sim/envs.py`
- Modify: `/Users/jerry/Code/orca_sim/tests/test_envs.py`

- [ ] **Step 1: Write failing tests for default and requested control periods.**

Require `BaseOrcaHandEnv` to expose `control_period_s`. Assert the existing default equals `model.opt.timestep * 5`. Instantiate the task with `control_period_s=0.08` and assert:

```python
assert env.frame_skip == 40
assert env.control_period_s == pytest.approx(0.08)
```

Also require a `ValueError` for non-positive periods and for periods smaller than half a simulator timestep, which round to zero steps.

- [ ] **Step 2: Run the focused tests and confirm missing-argument/property failures.**

```bash
uv run pytest tests/test_envs.py -q
```

- [ ] **Step 3: Resolve `frame_skip` after loading the MuJoCo model.**

Extend the base constructor without changing existing call sites:

```python
def __init__(
    self,
    scene_file: str,
    version: str | None = None,
    frame_skip: int = 5,
    render_mode: str | None = None,
    control_period_s: float | None = None,
) -> None:
```

Validate `frame_skip > 0`. Load the model, then if `control_period_s` is provided compute:

```python
resolved_steps = int(np.floor(control_period_s / model.opt.timestep + 0.5))
```

Reject `resolved_steps <= 0`; otherwise assign it to `self.frame_skip`. Set `self.control_period_s = float(self.model.opt.timestep * self.frame_skip)` for every environment.

- [ ] **Step 4: Run all base environment tests and commit.**

```bash
uv run pytest tests/test_envs.py tests/test_versions.py -q
git diff --check
git add src/orca_sim/envs.py tests/test_envs.py
git commit -m "Add physical control period configuration"
```

---

## Task 3: Add Full-Orientation Goal Modes and Target Policies

**Files:**

- Modify: `/Users/jerry/Code/orca_sim/src/orca_sim/task_envs.py`
- Create: `/Users/jerry/Code/orca_sim/tests/test_cube_orientation_goals.py`
- Modify: `/Users/jerry/Code/orca_sim/tests/test_registry.py`

- [ ] **Step 1: Write failing constructor and reset-info tests.**

Add tests for `goal_mode="cube_orientation"` with each supported policy and rejection of unknown modes/policies. On reset require normalized arrays:

```python
assert info["target_quat"].shape == (4,)
assert info["relative_target_quat"].shape == (4,)
assert info["relative_target_quat"][0] >= 0.0
assert info["orientation_error_rad"] == pytest.approx(
    quaternion_angular_error(info["cube_quat"], info["target_quat"])
)
```

For `face_direction`, keep the registry observation shape `(54,)`. For `cube_orientation`, require raw task observation shape `(58,)`: existing base 47 values plus cube normal 3, target quaternion 4, relative quaternion 4.

- [ ] **Step 2: Run focused tests and confirm constructor failures.**

```bash
uv run pytest tests/test_cube_orientation_goals.py tests/test_registry.py -q
```

- [ ] **Step 3: Extend the task constructor and state without changing defaults.**

Add keyword arguments:

```python
goal_mode: str = "face_direction"
target_policy: str = "cube_orientation_bag"
hand_mount_body_name: str = "right_mount"
control_period_s: float | None = None
max_task_duration_s: float | None = None
success_hold_duration_s: float | None = None
```

Use the orientation utility module instead of introducing another quaternion convention. Resolve `self._hand_mount_body_id`, cache the mount's nominal world rotation after `mj_forward`, and call `super(..., control_period_s=control_period_s)`.

- [ ] **Step 4: Implement fixed and random quarter-turn target issuance.**

Add one dispatcher:

```python
def _draw_next_orientation_target(
    self,
    current_quat: np.ndarray,
) -> tuple[np.ndarray, float]: ...
```

For `fixed_quarter_turn`, rotate `+pi/2` about nominal hand-frame X from the nominal initial cube quaternion. For `random_quarter_turn`, uniformly sample one of six signed hand axes and left-multiply the world-frame delta onto the current quaternion. Store the issued geodesic separation as `target_rotation_angle_rad`.

- [ ] **Step 5: Implement the 24-orientation shuffled bag.**

Compose each hand-frame cube rotation with the nominal cube orientation. When refilling the bag, skip a target within `success_tolerance_rad` of the current orientation and avoid the immediately preceding target. Test two consecutive bags for valid members and no adjacent repeats.

- [ ] **Step 6: Implement mode-specific observations and info.**

Keep all existing face fields. In orientation mode append `target_quat` and canonical `relative_target_quat`, use full quaternion error for `orientation_error_rad`, and expose `target_rotation_angle_rad`. `completed_target_quat` and `completed_target_rotation_angle_rad` must be set only on a completion event and otherwise be `None`.

- [ ] **Step 7: Run goal, registry, and legacy target-bag tests; commit.**

```bash
uv run pytest tests/test_cube_orientation_goals.py tests/test_registry.py tests/test_drop_penalty.py -q
git diff --check
git add src/orca_sim/task_envs.py tests/test_cube_orientation_goals.py tests/test_registry.py
git commit -m "Add full cube orientation goals"
```

---

## Task 4: Add Angular Progress and Duration-Based Task Events

**Files:**

- Modify: `/Users/jerry/Code/orca_sim/src/orca_sim/task_envs.py`
- Modify: `/Users/jerry/Code/orca_sim/tests/test_cube_orientation_goals.py`
- Modify: `/Users/jerry/Code/orca_sim/tests/test_drop_penalty.py`

- [ ] **Step 1: Write failing tests for duration conversion and diagnostics.**

At `control_period_s=0.08`, require `max_task_duration_s=8.0` to resolve to 100 steps and `success_hold_duration_s=0.16` to resolve to 2 steps. Assert reset/step info includes:

```python
assert info["control_period_s"] == pytest.approx(0.08)
assert info["max_task_duration_s"] == pytest.approx(8.0)
assert info["task_elapsed_s"] == pytest.approx(info["task_elapsed_steps"] * 0.08)
assert info["stable_success_duration_s"] == pytest.approx(
    info["stable_success_steps"] * 0.08
)
```

Reject non-positive durations. Preserve legacy step values when duration options are `None`.

- [ ] **Step 2: Run focused tests and confirm failures.**

```bash
uv run pytest tests/test_cube_orientation_goals.py tests/test_drop_penalty.py -q
```

- [ ] **Step 3: Convert duration overrides after the control period is resolved.**

Use a positive ceiling conversion so the physical condition is never shorter than requested:

```python
def _duration_to_steps(duration_s: float, period_s: float) -> int:
    return max(1, int(np.ceil(duration_s / period_s - 1e-12)))
```

Assign resolved step counts to the existing `max_task_steps` and `success_hold_steps` fields, so all legacy event logic shares one code path. Expose resolved physical durations from those counts.

- [ ] **Step 4: Write failing angular-progress and target-switch tests.**

Set `_previous_orientation_error_rad=1.0` and `_step_orientation_error_rad=0.8`; with scale 5 and step penalty 0.01 require reward `0.99`. Add exact success bonus and drop penalty checks. Complete a target and assert the next reward baseline is reset to the new goal's current error, so the switch itself contributes zero progress.

- [ ] **Step 5: Implement `angular_progress` and full-orientation success.**

Allow `reward_mode in {"absolute", "progress", "angular_progress"}`. Orientation success uses angular error instead of face alignment while retaining height, speed, hold, drop-first, sequence, and timeout rules. Track current/previous error in the same places as current/previous alignment.

- [ ] **Step 6: Run simulator task tests and commit.**

```bash
uv run pytest tests/test_cube_orientation_goals.py tests/test_drop_penalty.py tests/test_registry.py -q
git diff --check
git add src/orca_sim/task_envs.py tests/test_cube_orientation_goals.py tests/test_drop_penalty.py
git commit -m "Add timed angular cube task rewards"
```

---

## Task 5: Add the 47-Value State Environment Adapter

**Files:**

- Create: `/Users/jerry/Code/orca_train/src/orca_train/state_env.py`
- Create: `/Users/jerry/Code/orca_train/tests/test_state_env.py`

- [ ] **Step 1: Build a fake cube environment and write failing observation tests.**

The fake must expose 17 actuator joints, measured `qpos`, measured actuator `qvel`, cube info, a known `right_mount` transform, action ROM, and reset/step info. Require:

```python
observation, info = env.reset(seed=7)
assert observation["state"].shape == (47,)
assert observation["state"].dtype == np.float32
assert env.observation_space.contains(observation)
```

Assert exact slice ordering: `[0:17]` qpos, `[17:34]` qvel, `[34:37]` mount-frame cube position, `[37:40]` linear velocity, `[40:43]` angular velocity, `[43:47]` relative quaternion.

- [ ] **Step 2: Run the focused tests and confirm import failure.**

```bash
uv run pytest tests/test_state_env.py -q
```

- [ ] **Step 3: Implement construction and pure observation assembly.**

Create:

```python
class OrcaStateCubeEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    def __init__(
        self,
        *,
        env: gym.Env | None = None,
        version: str = "v2",
        max_delta_degrees: float = 3.0,
        fixed_joint_names: tuple[str, ...] = (),
        joint_velocity_limit: float = 10.0,
        workspace_radius: float = 0.25,
        linear_velocity_limit: float = 2.0,
        angular_velocity_limit: float = 20.0,
        **task_kwargs: Any,
    ) -> None: ...
```

When `env is None`, construct `OrcaHandRightCubeOrientation(goal_mode="cube_orientation", ...)`. Resolve actuator qpos and dof indices. Transform world vectors with the mount world rotation transpose; transform position after subtracting mount position. Scale and clip every component to `[-1, 1]`. Read `relative_target_quat` from task info and never call task status methods from `_observation`.

- [ ] **Step 4: Add and satisfy relative-action/fixed-joint tests.**

Reuse the visual wrapper rule: active targets are `measured_qpos + clipped_action * max_delta_radians`, clipped to ROM; fixed joints retain their reset target. Confirm the second step starts from changed measured qpos, not the prior command.

- [ ] **Step 5: Run focused tests and commit.**

```bash
uv run pytest tests/test_state_env.py -q
git diff --check
git add src/orca_train/state_env.py tests/test_state_env.py
git commit -m "Add state cube training environment"
```

---

## Task 6: Add Vector Replay and a Feed-Forward State Agent

**Files:**

- Modify: `/Users/jerry/Code/orca_train/src/orca_train/replay.py`
- Create: `/Users/jerry/Code/orca_train/src/orca_train/state_agent.py`
- Create: `/Users/jerry/Code/orca_train/tests/test_state_agent.py`
- Modify: `/Users/jerry/Code/orca_train/tests/test_drqv2.py`

- [ ] **Step 1: Write failing vector replay allocation and sampling tests.**

Add `StateReplayBatch` and `StateReplayBuffer`. Require contiguous float32 arrays for state, action, reward, discount, and next state. Validate public shapes on `add`, reject sampling more items than stored, and return device tensors.

- [ ] **Step 2: Run replay tests and confirm missing-class failure.**

```bash
uv run pytest tests/test_state_agent.py -q
```

- [ ] **Step 3: Implement the vector replay without changing image replay.**

Use interfaces parallel to `ReplayBuffer`:

```python
class StateReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, seed: int = 0): ...
    def add(self, observation, action, reward, discount, next_observation) -> None: ...
    def sample(self, batch_size: int, device: torch.device) -> StateReplayBatch: ...
    def __len__(self) -> int: ...
```

The observation boundary is `{"state": np.ndarray}` so the existing `NStepAccumulator` can be reused unchanged.

- [ ] **Step 4: Write failing state agent action/update tests.**

Require:

```python
config = StateAgentConfig(hidden_dim=32, batch_size=2, update_every_steps=1)
agent = StateAgent(state_dim=47, action_dim=16, device="cpu", config=config)
action = agent.act({"state": np.zeros(47, np.float32)}, step=0, eval_mode=True)
assert action.shape == (16,)
assert np.all(np.abs(action) <= 1.0)
```

Fill replay, call one update, and require finite `critic_loss`, `actor_loss`, `q`, and `target_q`.

- [ ] **Step 5: Implement the state agent with the existing deterministic twin-Q core.**

Import and reuse `Actor`, `Critic`, `actor_loss_from_q_values`, `sample_noisy_action`, and `schedule`. Add:

```python
@dataclass(frozen=True)
class StateAgentConfig:
    hidden_dim: int = 1024
    learning_rate: float = 1e-4
    critic_target_tau: float = 0.01
    stddev_schedule: str = "linear(1.0,0.1,500000)"
    stddev_clip: float = 0.3
    batch_size: int = 256
    update_every_steps: int = 2
```

Do not add an encoder or image augmentation.

- [ ] **Step 6: Add checkpoint metadata and compatibility tests.**

Save `observation_mode="state"`, `state_dim`, `action_dim`, config, step, model states, and optimizer states. On load, compare mode and dimensions before loading tensors and raise `ValueError` naming the mismatched field. Test round-trip and all three compatibility errors.

- [ ] **Step 7: Run state plus visual-agent regression tests and commit.**

```bash
uv run pytest tests/test_state_agent.py tests/test_drqv2.py -q
git diff --check
git add src/orca_train/replay.py src/orca_train/state_agent.py tests/test_state_agent.py tests/test_drqv2.py
git commit -m "Add feed-forward state cube agent"
```

---

## Task 7: Add Curriculum Resolution, Evaluation Metrics, and State Training

**Files:**

- Create: `/Users/jerry/Code/orca_train/src/orca_train/train_state.py`
- Create: `/Users/jerry/Code/orca_train/tests/test_train_state.py`
- Modify: `/Users/jerry/Code/orca_train/pyproject.toml`

- [ ] **Step 1: Write failing preset resolution tests.**

Create immutable `StateTrainConfig` and `resolve_preset(config)`. Require exact preset values:

```python
single_goal = {"target_policy": "fixed_quarter_turn", "target_sequence_length": 1,
               "success_tolerance_rad": 0.4}
right_angle = {"target_policy": "random_quarter_turn", "target_sequence_length": 1,
               "success_tolerance_rad": 0.4}
multi_goal = {"target_policy": "cube_orientation_bag", "target_sequence_length": 20,
              "success_tolerance_rad": np.deg2rad(15.0)}
```

Every preset resolves `control_period_s=0.08`, `max_task_duration_s=8.0`, `success_hold_duration_s=0.16`, and `reward_mode="angular_progress"`. Use `None` for CLI-overridable preset fields in the input config so explicit values win. Reject unknown preset names.

- [ ] **Step 2: Run focused tests and confirm import failure.**

```bash
uv run pytest tests/test_train_state.py -q
```

- [ ] **Step 3: Implement deterministic orientation evaluation aggregation.**

Add `evaluate_state(agent, env, episodes, seed)` and test fixed reset seeds. Aggregate return, full success, mean/median/total targets, drop/timeout rates, mean/median seconds per completed target, mean final error, and mean best error. Track per-episode best error from reset onward. Bucket each completed target's issued angle to `90`, `120`, or `180` degrees within a `1e-5` rad tolerance and report count, completion rate, and mean completion seconds for buckets that occur.

- [ ] **Step 4: Implement a fake-environment training smoke test first.**

Use a 47-state, 16-action two-step fake environment. Run four training steps with a batch size of two, evaluation/checkpoint intervals of two, and CPU. Require `config.json`, update/evaluation/train events in `metrics.jsonl`, `checkpoint_4.pt`, successful reload, and closure of train/eval environments.

- [ ] **Step 5: Implement the state training loop.**

Mirror the proven visual loop but use `OrcaStateCubeEnv`, `StateReplayBuffer`, `StateAgent`, and `evaluate_state`. Persist the fully resolved `StateTrainConfig` before environment construction. Reuse `NStepAccumulator`, `select_device`, and `write_metric` from `train.py`. Pass all resolved task values explicitly into the adapter.

- [ ] **Step 6: Implement CLI parsing with explicit override precedence.**

Add `--preset {single_goal,right_angle,multi_goal}` plus shared training options and nullable overrides for all preset fields. Convert `--success-tolerance-degrees` to radians only after parsing and reject simultaneous radian/degree overrides. Register:

```toml
orca-train-state = "orca_train.train_state:main"
```

- [ ] **Step 7: Run focused training tests and commit.**

```bash
uv run pytest tests/test_train_state.py -q
git diff --check
git add src/orca_train/train_state.py tests/test_train_state.py pyproject.toml
git commit -m "Add state cube curriculum trainer"
```

---

## Task 8: Document, Smoke-Test, and Verify Both Repositories

**Files:**

- Modify: `/Users/jerry/Code/orca_train/README.md`
- Modify if needed for test correctness only: files changed in Tasks 1-7

- [ ] **Step 1: Document the separate state baseline and first run.**

Add the observation layout, preset table, checkpoint compatibility note, and this exact first-run command:

```bash
cd /Users/jerry/Code/orca_train
uv run orca-train-state \
  --preset single_goal \
  --device mps \
  --total-steps 1000000 \
  --output-dir runs/state_cube_single_goal_seed1
```

State clearly that the smoke run validates plumbing, not learned manipulation performance.

- [ ] **Step 2: Run the complete simulator suite.**

```bash
cd /Users/jerry/Code/orca_sim
uv run pytest tests/
git diff --check
```

Expected: all legacy and new tests pass.

- [ ] **Step 3: Run the complete training suite.**

```bash
cd /Users/jerry/Code/orca_train
uv run pytest
git diff --check
```

Expected: all visual and state tests pass.

- [ ] **Step 4: Run a real-environment short smoke command with no long training.**

```bash
uv run orca-train-state \
  --preset single_goal \
  --device cpu \
  --total-steps 4 \
  --seed-steps 2 \
  --batch-size 2 \
  --eval-every-steps 2 \
  --eval-episodes 1 \
  --checkpoint-every-steps 2 \
  --replay-capacity 32 \
  --hidden-dim 32 \
  --output-dir /tmp/orca_state_cube_smoke
```

Inspect the resolved `config.json`, final evaluation event, and checkpoint metadata. Do not interpret four steps as a policy-quality result.

- [ ] **Step 5: Commit documentation and any verified corrections.**

```bash
git add README.md
git commit -m "Document state cube baseline training"
```

- [ ] **Step 6: Record final evidence and repository state.**

For each repository, capture `git status --short --branch` and `git log -5 --oneline`. Report test counts, smoke command outcome, branch names, and exact first long-run command. Do not push or open PRs without a separate user request.

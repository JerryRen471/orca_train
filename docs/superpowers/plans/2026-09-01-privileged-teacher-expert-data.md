# Privileged Teacher and Expert Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a 131-observation privileged PPO teacher with stage-balanced resets, validate it on all six task stages, and collect at least 600 successful exact-environment trajectory segments.

**Architecture:** Extend the existing curriculum with an episode-fixed balanced mode that samples all six task stages uniformly. Reuse the production PPO runner with a teacher network profile whose policy and value keys are both `privileged_state`. A GPU collector rolls out the frozen teacher separately on each fixed stage, accepts only first-success/no-drop segments, and writes them through the immutable dataset schema from the public-BC phase.

**Tech Stack:** Python 3.12, `uv`, JAX 0.6.2, Flax 0.11.2, Optax 0.2.6, Brax 0.14.2, MuJoCo 3.6.0, MuJoCo Warp/MJX, MuJoCo Playground v0.2.0, NumPy, pytest, Spark GB10 GPU.

**Spec:** `docs/superpowers/specs/2026-09-01-expert-guided-cube-reorientation-design.md`

## Global Constraints

- Complete `2026-09-01-public-orca-bc-diagnostic.md` first; this plan consumes `expert_schema.py` and `policy_suite.py`.
- Use the same `feature/expert-guided-cube-reorientation` worktree and `uv` environment.
- The teacher policy and value networks both consume exactly 131 dimensions.
- The teacher emits the unchanged normalized 16-dimensional relative action; the wrist remains fixed to `1e-8` radians.
- Preserve the Leap reward, action scale `0.5`, network widths `(512, 256, 128)`, observation noise, domain randomization, termination, `ctrl_dt=0.05`, and `sim_dt=0.01`.
- Each reset chooses one of six stages with probability `1/6`; the chosen stage cannot change before episode reset.
- Run 2 million steps first. Continue to at most 20 million only if a turning stage improves and aggregate drop rate increases by no more than one percentage point.
- Do not label a teacher as expert unless 256-seed evaluation reaches hold success `>=0.99` and every turning-stage success `>=0.80`.
- Store at least 100 accepted segments per stage. Failed or rejected rollouts remain diagnostic records and never enter the expert training split.

## File map

### Create

- `src/orca_train/teacher_run.py` — teacher-specific PPO configuration, run gate, checkpoint ranking, and CLI.
- `src/orca_train/expert_collect.py` — exact-environment rollout collection, success gates, SO(3) coverage, and CLI.
- `tests/test_teacher_run.py`
- `tests/test_expert_collect.py`

### Modify

- `src/orca_train/playground_curriculum.py` — balanced mode and uniform stage sampling.
- `src/orca_train/playground_env.py` — balanced reset, within-stage goal refresh, and canonical actor components.
- `src/orca_train/playground_config.py` — teacher network profile and explicit 2M/20M budgets.
- `src/orca_train/playground_run.py` — run-role manifest fields, raw restore parameters, and teacher-compatible classification.
- `src/orca_train/policy_suite.py` — teacher gate and checkpoint ranking.
- `pyproject.toml` — teacher and collection commands.
- `README.md` — teacher smoke, continuation, evaluation, and collection workflow.
- `tests/test_playground_curriculum.py`
- `tests/test_playground_env.py`
- `tests/test_playground_config.py`
- `tests/test_playground_run.py`
- `tests/test_policy_suite.py`

---

### Task 1: Add uniform episode-fixed stage sampling

**Files:**
- Modify: `src/orca_train/playground_curriculum.py`
- Modify: `tests/test_playground_curriculum.py`

**Interfaces:**
- Consumes: a JAX PRNG key.
- Produces: `CURRICULUM_MODE_BALANCED`, `sample_balanced_stage_index(rng) -> jax.Array`, and updated `validate_curriculum_mode`.

- [ ] **Step 1: Write failing balanced-mode tests**

```python
from orca_train.playground_curriculum import (
    CURRICULUM_MODE_BALANCED,
    sample_balanced_stage_index,
    validate_curriculum_mode,
)


def test_balanced_mode_is_valid_and_samples_every_stage() -> None:
    validate_curriculum_mode(CURRICULUM_MODE_BALANCED, 5)
    keys = jax.random.split(jax.random.PRNGKey(20260901), 60_000)
    sampled = np.asarray(jax.vmap(sample_balanced_stage_index)(keys))
    counts = np.bincount(sampled, minlength=6)
    assert sampled.min() == 0
    assert sampled.max() == 5
    np.testing.assert_allclose(counts / counts.sum(), np.full(6, 1 / 6), atol=0.01)


def test_balanced_stage_sampler_is_jittable() -> None:
    value = jax.jit(sample_balanced_stage_index)(jax.random.PRNGKey(3))
    assert value.shape == ()
    assert value.dtype == jnp.int32
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_playground_curriculum.py -v`

Expected: FAIL because balanced mode is undefined.

- [ ] **Step 3: Implement the minimal sampler**

```python
CURRICULUM_MODE_BALANCED = "balanced"


def sample_balanced_stage_index(rng: jax.Array) -> jax.Array:
    return jax.random.randint(
        rng,
        (),
        minval=0,
        maxval=len(ORCA_REORIENT_CURRICULUM),
        dtype=jp.int32,
    )
```

Allow `fixed`, `online`, and `balanced` in validation. Promotion thresholds remain validated for checkpoint compatibility even though balanced mode never calls promotion.

- [ ] **Step 4: Run tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_playground_curriculum.py -v`

Expected: PASS.

```bash
git add src/orca_train/playground_curriculum.py tests/test_playground_curriculum.py
git commit -m "Add balanced reorientation stage sampler"
```

---

### Task 2: Integrate balanced resets and within-stage goal refresh

**Files:**
- Modify: `src/orca_train/playground_env.py`
- Modify: `tests/test_playground_env.py`

**Interfaces:**
- Consumes: `curriculum_mode="balanced"`.
- Produces: episode-fixed `info["curriculum_stage_index"]`, same-stage goal refresh after success, and `canonical_actor_components(data, info)`.

- [ ] **Step 1: Write failing balanced reset tests**

```python
def test_balanced_reset_uses_sampled_stage_and_keeps_it_for_episode(validated_grasp) -> None:
    environment = cpu_environment(validated_grasp, curriculum_mode="balanced")
    keys = jax.random.split(jax.random.PRNGKey(4), 1_000)
    stages = np.asarray(jax.vmap(lambda key: environment.reset(key).info["curriculum_stage_index"])(keys))
    assert set(stages.tolist()) == set(range(6))

    state = environment.reset(jax.random.PRNGKey(5))
    initial_stage = int(state.info["curriculum_stage_index"])
    for _ in range(20):
        state = environment.step(state, jnp.zeros(16))
        assert int(state.info["curriculum_stage_index"]) == initial_stage
```

- [ ] **Step 2: Write failing goal-refresh tests**

Force success with a no-physics stub. For stages 1 through 4, assert the refreshed goal remains within that stage's angle range from the current cube. For hold, assert the goal remains the current orientation. For full SO(3), assert `goal_quat_dquat` follows the existing Leap moving-goal path. In every case, assert the stage index is unchanged.

- [ ] **Step 3: Implement balanced reset selection**

Split a `stage_key` in `reset`. Resolve the initial index as:

```python
if self._curriculum_mode == CURRICULUM_MODE_ONLINE:
    initial_stage_index = jp.asarray(0, dtype=jp.int32)
elif self._curriculum_mode == CURRICULUM_MODE_BALANCED:
    initial_stage_index = sample_balanced_stage_index(stage_key)
else:
    initial_stage_index = jp.asarray(self._curriculum_stage_index, dtype=jp.int32)
```

Use `sample_goal_for_stage_index` for the reset goal and store the scalar index in `info` and metrics.

- [ ] **Step 4: Implement balanced success handling**

When balanced and stage is 1 through 4, split a new goal key, sample from the same stage around the current cube orientation, and install it only when `success` is true. Hold never changes its goal. Full SO(3) uses the existing `goal_quat_dquat` update. Do not call `advance_online_curriculum` in balanced mode.

- [ ] **Step 5: Add canonical actor-component parity tests**

```python
def test_canonical_components_rebuild_noise_free_actor_state(validated_grasp) -> None:
    environment = cpu_environment(validated_grasp, obs_noise_level=0.0)
    state = environment.reset(jax.random.PRNGKey(6))
    components = environment.canonical_actor_components(state.data, state.info)
    rebuilt = build_actor_observation(jax.tree.map(np.asarray, components))
    np.testing.assert_allclose(rebuilt, np.asarray(state.obs["state"]), atol=1e-6)
```

- [ ] **Step 6: Implement `canonical_actor_components`**

Return clean joint positions, previous active motor targets, palm position, clean cube position/quaternion, goal quaternion, and previous action. Refactor `_get_obs` only enough to reuse the same clean fields; keep all existing noise draws and 57/131 slice order unchanged.

- [ ] **Step 7: Run environment tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_playground_curriculum.py tests/test_playground_env.py -v`

Expected: PASS, including all existing fixed and online curriculum tests.

```bash
git add src/orca_train/playground_env.py tests/test_playground_env.py
git commit -m "Use balanced stages in MJX reorientation"
```

---

### Task 3: Add the privileged teacher PPO profile and run provenance

**Files:**
- Modify: `src/orca_train/playground_config.py`
- Modify: `src/orca_train/playground_run.py`
- Create: `src/orca_train/teacher_run.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_playground_config.py`
- Modify: `tests/test_playground_run.py`
- Create: `tests/test_teacher_run.py`

**Interfaces:**
- Consumes: the existing ORCA PPO config and validated grasp/reset inputs.
- Produces: `TEACHER_SMOKE_TIMESTEPS`, `TEACHER_MAX_TIMESTEPS`, `teacher_ppo_config`, manifest `run_role/policy_obs_key/value_obs_key`, generalized `train_playground(grasp_artifact, validation_report, reset_validation_report, output_dir, ppo_config_override, run_role, initialization_kind, initialization_sha256)`, `train_teacher`, and `orca-train-privileged-teacher`.

- [ ] **Step 1: Write failing teacher config tests**

```python
def test_teacher_config_changes_only_policy_observation_and_budget() -> None:
    student = orca_ppo_config()
    teacher = teacher_ppo_config(num_timesteps=2_000_000)
    assert teacher.network_factory.policy_obs_key == "privileged_state"
    assert teacher.network_factory.value_obs_key == "privileged_state"
    assert teacher.num_timesteps == 2_000_000
    student.num_timesteps = 2_000_000
    student.network_factory.policy_obs_key = "privileged_state"
    assert teacher.to_dict() == student.to_dict()
```

Also reject budgets other than `2_000_000` and `20_000_000` from the teacher CLI.

- [ ] **Step 2: Implement teacher config constants**

```python
TEACHER_SMOKE_TIMESTEPS = 2_000_000
TEACHER_MAX_TIMESTEPS = 20_000_000


def teacher_ppo_config(*, num_timesteps: int) -> config_dict.ConfigDict:
    if num_timesteps not in {TEACHER_SMOKE_TIMESTEPS, TEACHER_MAX_TIMESTEPS}:
        raise ValueError("teacher budget must be 2M or 20M steps")
    config = orca_ppo_config()
    config.num_timesteps = num_timesteps
    config.network_factory.policy_obs_key = "privileged_state"
    config.network_factory.value_obs_key = "privileged_state"
    return config
```

- [ ] **Step 3: Write failing run-role manifest tests**

Create teacher and student manifests. Require new writes to use schema version 4, `run_role` in `{"teacher", "student"}`, exact policy/value observation keys, balanced curriculum for teacher, and classification `teacher_candidate` for both 2M and 20M teacher runs. Assert schema 3 baseline manifests are migrated read-only to role `student`, policy key `state`, value key `privileged_state`, and initialization `ppo_legacy`; schemas below 3 fail explicitly.

- [ ] **Step 4: Extend the manifest and resume contract**

Add fields:

```python
run_role: str
policy_obs_key: str
value_obs_key: str
initialization_kind: str
initialization_sha256: str | None
```

Include all fields in `_RESUME_FIELDS`. `classify_run` returns `teacher_candidate` only for `run_role="teacher"`, balanced curriculum, privileged policy/value keys, and an allowed teacher budget. It never returns `accepted`.

Keep schema-3 migration inside `RunManifest.read`; do not rewrite the original
20M manifest on disk. New teacher/student resume targets must write schema 4,
while evaluation and public BC warm start may consume the migrated baseline.

- [ ] **Step 5: Implement the teacher runner as a thin wrapper**

Extend `train_playground` with explicit keyword-only parameters:

```python
ppo_config_override: config_dict.ConfigDict | None = None,
run_role: str = "student",
initialization_kind: str = "random",
initialization_sha256: str | None = None,
```

Use a copied override instead of constructing `orca_ppo_config()` when it is
present, and bind the four values into the manifest before writing any output.
`train_teacher` constructs `teacher_ppo_config`, then calls the generalized
runner with `run_role="teacher"`, `curriculum_mode="balanced"`, and no actor
initialization. The CLI requires `--num-timesteps {2000000,20000000}` and
accepts `--resume-checkpoint` only for the 20M continuation.

- [ ] **Step 6: Prove both networks receive 131 dimensions**

Monkeypatch `ppo_networks.make_ppo_networks` in a unit test, call `build_train_fn`, and assert the observation-size mapping contains 57/131 while `policy_obs_key` and `value_obs_key` are both `privileged_state`. Add a negative test that a teacher config with policy key `state` is rejected before environment construction.

- [ ] **Step 7: Run tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_playground_config.py tests/test_playground_run.py tests/test_teacher_run.py -v`

Expected: PASS.

```bash
git add pyproject.toml src/orca_train/playground_config.py src/orca_train/playground_run.py src/orca_train/teacher_run.py tests/test_playground_config.py tests/test_playground_run.py tests/test_teacher_run.py
git commit -m "Add privileged PPO teacher runner"
```

---

### Task 4: Add teacher stage gates and checkpoint ranking

**Files:**
- Modify: `src/orca_train/policy_suite.py`
- Modify: `src/orca_train/teacher_run.py`
- Modify: `tests/test_policy_suite.py`
- Modify: `tests/test_teacher_run.py`

**Interfaces:**
- Consumes: a teacher `PolicySuiteReport` and optional initial report.
- Produces: `teacher_is_expert`, `teacher_smoke_allows_continuation`, `rank_teacher_reports`, and `teacher-gate.json`.

- [ ] **Step 1: Write failing expert-gate tests**

```python
def test_teacher_expert_gate_requires_hold_99_and_every_turn_80() -> None:
    passing = suite_report(hold=0.99, turns=[0.80] * 5, full_drop=0.05)
    assert teacher_is_expert(passing)
    assert not teacher_is_expert(suite_report(hold=0.98, turns=[0.90] * 5))
    assert not teacher_is_expert(suite_report(hold=1.0, turns=[0.80, 0.80, 0.79, 0.90, 0.90]))
```

Require zero non-finite episodes and wrist drift at most `1e-8` in all stages.

- [ ] **Step 2: Write failing continuation-gate tests**

The 2M report may continue only if at least one turning-stage success is strictly higher than the initial untrained teacher and aggregate drop is no more than `initial + 0.01`. Assert equality on every turning stage fails, `+0.0100001` drop fails, and a finite improvement passes.

- [ ] **Step 3: Implement deterministic ranking**

Rank by descending arithmetic mean of six success rates, then descending full-SO3 success, then ascending aggregate drop, then lexicographic checkpoint SHA-256. Store every candidate score in `teacher-gate.json`; do not delete losing checkpoints.

- [ ] **Step 4: Add CLI gate command**

Add `orca-gate-privileged-teacher` taking `--initial-report`, one or more `--candidate-report`, `--phase {smoke,expert}`, and `--output`. Return exit code 0 only when the requested gate passes and write the JSON report in both pass and fail cases.

- [ ] **Step 5: Run tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_policy_suite.py tests/test_teacher_run.py -v`

Expected: PASS.

```bash
git add pyproject.toml src/orca_train/policy_suite.py src/orca_train/teacher_run.py tests/test_policy_suite.py tests/test_teacher_run.py
git commit -m "Gate privileged teacher by stage metrics"
```

---

### Task 5: Collect first-success exact-environment segments

**Files:**
- Create: `src/orca_train/expert_collect.py`
- Create: `tests/test_expert_collect.py`
- Modify: `src/orca_train/playground_env.py`
- Modify: `tests/test_playground_env.py`

**Interfaces:**
- Consumes: a frozen teacher checkpoint, fixed stage, grasp inputs, seed range, batch size, and target accepted count.
- Produces: `TrajectoryDecision`, `collect_stage_segments`, `accepted_segment`, and per-stage episode files through `write_episode`.

- [ ] **Step 1: Write failing segment acceptance tests**

```python
@pytest.mark.parametrize(
    ("success", "dropped", "nonfinite", "wrist", "accepted", "reason"),
    [
        (True, False, False, 0.0, True, None),
        (False, False, False, 0.0, False, "no_success"),
        (True, True, False, 0.0, False, "dropped"),
        (True, False, True, 0.0, False, "nonfinite"),
        (True, False, False, 1.1e-8, False, "wrist_drift"),
    ],
)
def test_segment_gate(success, dropped, nonfinite, wrist, accepted, reason) -> None:
    decision = accepted_segment(
        success=success,
        dropped=dropped,
        nonfinite=nonfinite,
        max_wrist_drift_rad=wrist,
        actions=np.zeros((4, 16), np.float32),
    )
    assert decision.accepted is accepted
    assert decision.reason == reason
```

Add an out-of-range action case with reason `action_bounds`.

- [ ] **Step 2: Write a failing first-success slicing test**

Given a fake rollout whose success sequence is `[False, False, True, True]`, assert the saved segment contains exactly three transitions, uses the goal quaternion from before the successful step, and ignores the later transition. For hold, require exactly 40 stable steps.

- [ ] **Step 3: Expose exact collector fields without changing observations**

Add a method returning a JAX pytree with canonical actor components, canonical 57 state, 131 privileged state, cube pose/velocity, goal quaternion, orientation error, and wrist deviation for a given `data/info`. Unit-test that calling it does not mutate `info`, consume RNG, or alter `_get_obs` output.

- [ ] **Step 4: Implement batched fixed-stage rollout collection**

For each stage, construct `OrcaCubeReorient(curriculum_mode="fixed", curriculum_stage=stage)`, wrap it with the same domain randomization, and use `jax.lax.scan` for at most 1,000 policy steps (40 for hold). Record pre-action observation/goal, action, and post-step outcome. Transfer completed batches to host, slice first-success segments, write accepted episodes atomically, and append rejected seed/reason records to `rejections.jsonl`.

Each accepted `ExpertEpisode` must populate current/next canonical 57 states,
current/next 131 privileged states, and these transition-aligned extras:

```python
EXACT_REQUIRED_EXTRAS = (
    "cube_position",
    "cube_quaternion",
    "cube_linear_velocity",
    "cube_angular_velocity",
    "goal_quaternion",
    "orientation_error_rad",
    "reward",
    "success",
    "dropped",
    "terminated",
    "truncated",
    "goal_index",
)
```

Store reset seed, stage, domain-randomization parameters, source revisions,
teacher checkpoint hash, and environment/model/grasp fingerprints in episode
metadata.

- [ ] **Step 5: Add provenance mismatch tests**

Reject a teacher checkpoint unless its manifest has role `teacher`, policy/value key `privileged_state`, matching grasp/reset/model/active-joint hashes, balanced training mode, and an expert-passing gate report bound to that checkpoint SHA.

- [ ] **Step 6: Run collector tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_expert_collect.py tests/test_playground_env.py -v`

Expected: PASS with fake policies/environments; no GPU is required for unit tests.

```bash
git add src/orca_train/expert_collect.py src/orca_train/playground_env.py tests/test_expert_collect.py tests/test_playground_env.py
git commit -m "Collect verified teacher success segments"
```

---

### Task 6: Enforce per-stage counts and full-SO3 coverage

**Files:**
- Modify: `src/orca_train/expert_collect.py`
- Modify: `tests/test_expert_collect.py`

**Interfaces:**
- Consumes: accepted per-stage episode entries.
- Produces: `EXACT_REQUIRED_EXTRAS`, `rotation_coverage_cell`, `validate_expert_dataset`, immutable dataset manifest, and `collection-report.json`.

- [ ] **Step 1: Write failing SO(3) cell tests**

Construct relative quaternions at angles `pi/6`, `pi/2`, and `5*pi/6` with axes in all eight sign octants. Assert `rotation_coverage_cell` returns 24 unique `(angle_band, axis_octant)` pairs. Canonicalize the axis sign for the quaternion's shortest rotation and handle zero angle deterministically.

- [ ] **Step 2: Write failing dataset gate tests**

Assert validation fails with 99 episodes in any stage, fewer than four full-SO3 segments in any coverage cell, duplicate reset seeds, a teacher-training/evaluation seed, a missing `EXACT_REQUIRED_EXTRAS` field, a content-hash mismatch, or any privileged width other than 131. Assert exactly 100 per stage with four per full-SO3 cell plus four extra passes.

- [ ] **Step 3: Implement coverage and finalization**

Use angle bands `[0, pi/3)`, `[pi/3, 2*pi/3)`, and `[2*pi/3, pi]`. Encode the octant as three sign bits after normalizing the rotation axis. `validate_expert_dataset` verifies counts, coverage, independent seeds, all episode hashes, source checkpoint/gate hashes, environment fingerprints, and wrist/action/success constraints before setting `classification="expert"`.

- [ ] **Step 4: Add collection CLI**

Add `orca-collect-expert-trajectories` with required teacher checkpoint/gate/grasp/reset/output arguments, `--accepted-per-stage 100`, `--seed-start`, and `--batch-size`. Refuse an existing manifest, resume only from verified episode files plus the same resolved config, and stop when every stage and coverage cell passes.

- [ ] **Step 5: Run tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_expert_schema.py tests/test_expert_collect.py -v`

Expected: PASS.

```bash
git add pyproject.toml src/orca_train/expert_collect.py tests/test_expert_collect.py
git commit -m "Validate balanced expert trajectory coverage"
```

---

### Task 7: Execute the teacher gate and expert collection on Spark

**Files:**
- Modify: `README.md`
- Modify: `tests/test_teacher_run.py`
- Modify: `tests/test_expert_collect.py`

**Interfaces:**
- Consumes: committed source snapshots and validated reset inputs.
- Produces: initial/2M/optional-20M teacher reports, expert gate, teacher checkpoint, 600+ episode dataset, collection report, logs, and hashes.

- [ ] **Step 1: Document and test exact CLI parsing**

Require explicit output directories and seed lists. Assert teacher defaults are balanced mode, 8,192 environments, seed 1, and domain randomization enabled. Assert collection refuses fewer than 100 segments per stage.

- [ ] **Step 2: Run the complete local regression gate**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests -q`

Expected: all tests pass.

- [ ] **Step 3: Sync committed sources to a versioned Spark directory**

Create `/data/home/scv7454/run/orca_expert_guided_20260901` if absent. Sync committed `orca_train` and the exact matching `orca_sim` snapshot without `--delete`, write `source-revisions.json`, and verify remote SHA-256 for the lockfile, grasp artifact, reset report, and MJCF source.

- [ ] **Step 4: Run remote dependency and GPU smoke gates**

Run `uv sync --extra playground --extra expert --group dev`, the full CPU-marked test suite, a JAX GPU assertion, one batched Warp reset/step, and a shortened 8-environment PPO update. Require finite 57/131 observations, finite loss/gradients/parameters, all six reset stages represented, wrist drift `<=1e-8`, and a restorable checkpoint.

- [ ] **Step 5: Record an untrained teacher reference report**

Evaluate the initialized privileged policy with 256 fixed seeds per stage and write `teacher-initial-suite.json`. Bind its seed-list hash to every later gate.

- [ ] **Step 6: Run the 2-million-step teacher in a named persistent session**

Use output `runs/privileged_teacher_seed1_2m`, seed 1, 8,192 environments, balanced curriculum, and exactly 2,000,000 requested steps. Preserve stdout/stderr, metrics JSONL, manifests, all checkpoints, and session name in `remote-run.json`.

- [ ] **Step 7: Evaluate and apply the continuation gate**

Evaluate every saved candidate with 256 identical seeds per stage, rank them deterministically, and write `teacher-smoke-gate.json`. If the gate fails, stop and copy diagnostics back. If it passes but is not yet expert, resume the selected checkpoint into `runs/privileged_teacher_seed1_20m` with the 20,000,000 ceiling.

- [ ] **Step 8: Apply the expert gate and collect trajectories**

Evaluate the final ranked candidates, write `teacher-expert-gate.json`, and proceed only if hold is at least 99% and every turning stage is at least 80%, with finite metrics and wrist invariance. Collect at least 100 accepted segments per stage into `runs/expert_trajectories_seed1_v1`, validate all 24 full-SO3 coverage cells, and finalize the immutable expert manifest.

- [ ] **Step 9: Copy artifacts back and verify**

Copy teacher manifests/checkpoints, all suite/gate reports, logs, the immutable dataset, rejection log, and collection report to local `runs/expert_guided_20260901/`. Recompute hashes locally and compare them with the remote manifest before reporting the teacher phase complete.

- [ ] **Step 10: Commit documentation**

```bash
git add README.md tests/test_teacher_run.py tests/test_expert_collect.py
git commit -m "Document privileged teacher data generation"
```

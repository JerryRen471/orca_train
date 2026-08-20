# ORCA LeapCubeReorient PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, evaluate, and export an ORCA cube-reorientation policy using the pinned MuJoCo Playground `LeapCubeReorient` Brax PPO configuration and a 57/131-dimensional asymmetric observation pair.

**Architecture:** `orca_sim` owns the physical stable-grasp artifact, site definitions, preparation controller, CEM search, and CPU acceptance checks. `orca_train` owns a thin pinned-Playground adapter, an ORCA Warp/MJX environment, the six-stage curriculum, Brax PPO orchestration, evaluation, and export. Full search and the exact 8192-environment, 200-million-step run execute on Spark after local and remote smoke gates pass.

**Tech Stack:** Python 3.11/3.12, `uv`, MuJoCo 3.6+, MuJoCo MJX/Warp, JAX CUDA 12, MuJoCo Playground v0.2.0, Brax PPO, Gymnasium, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-leap-cube-reorient-ppo-design.md`

## Global Constraints

- Work on isolated feature branches/worktrees; never commit directly to `main`.
- Use `uv` for all local Python dependency, test, and command execution.
- Pin Playground to tag `v0.2.0`; record the resolved dependency revision and lock hash.
- Final PPO config: 200,000,000 steps, 20 evals, 8192 envs, batch 256, unroll 40, 32 minibatches, 4 updates/batch, discount 0.99, learning rate `3e-4`, entropy `1e-2`, networks `(512, 256, 128)`.
- Final environment config: `ctrl_dt=0.05`, `sim_dt=0.01`, `action_scale=0.5`, `action_repeat=1`, `episode_length=1000`, `success_threshold=0.1`, `history_len=1`, implementation `warp`.
- Fix `right_wrist`; PPO acts on exactly 16 finger actuators.
- Actor observation is exactly 57 dimensions; privileged critic observation is exactly 131 dimensions.
- Do not add a custom grasp-quality reward, residual-action envelope, invisible support, weld, or gravity modification.
- Grasp gate: 100/100 nominal holds and at least 95% held-out perturbed stability before PPO.
- Final gate: full SO(3), at least 80% held-out success, at most 5% held-out drop rate.
- Reduced steps or environment counts are diagnostic only and cannot produce an accepted model.

## File map

### `orca_sim`

- Modify `src/orca_sim/models/v2/mjcf/orcahand_right_body.xml` — named palm and five fingertip sites.
- Modify `src/orca_sim/scenes/v2/scene_right_cube_orientation.xml` — visible goal mocap body and task sensor names.
- Create `src/orca_sim/grasping.py` — artifact schema/hash, placement, closing trajectory, stability gate, preparation.
- Create `src/orca_sim/grasp_search.py` — deterministic CEM, perturbation evaluation, search/validate CLI.
- Modify `src/orca_sim/task_envs.py` — opt-in stable-grasp preparation and structured diagnostics.
- Modify `src/orca_sim/__init__.py` — public grasp exports.
- Modify `pyproject.toml` — `orca-grasp` command.
- Create `tests/test_grasping.py`, `tests/test_grasp_reset.py`, `tests/test_grasp_search.py`.

### `orca_train`

- Modify `pyproject.toml` and `uv.lock` — pinned optional Playground dependency and new commands.
- Create `src/orca_train/playground_constants.py` — model names, active ordering, schema versions.
- Create `src/orca_train/playground_config.py` — exact environment/PPO config and parity checks.
- Create `src/orca_train/playground_curriculum.py` — six stages and deterministic promotion.
- Create `src/orca_train/playground_env.py` — ORCA Warp/MJX environment, asymmetric observations, reward, randomization.
- Create `src/orca_train/playground_run.py` — manifests, Brax PPO training, checkpointing, metrics.
- Create `src/orca_train/playground_evaluate.py` — held-out evaluation and acceptance report.
- Create `src/orca_train/playground_export.py` — policy-only export and deterministic parity check.
- Create `src/orca_train/playground_video.py` — accepted-checkpoint rollout rendering.
- Create `tests/test_playground_constants.py`, `tests/test_playground_config.py`, `tests/test_playground_curriculum.py`, `tests/test_playground_env.py`, `tests/test_playground_randomization.py`, `tests/test_playground_run.py`, `tests/test_playground_export.py`.
- Modify `README.md` — exact local, Spark, resume, evaluate, export, and render commands.

---

### Task 1: Add palm and fingertip model sites

**Repository:** `orca_sim`

**Files:**
- Modify: `src/orca_sim/models/v2/mjcf/orcahand_right_body.xml`
- Modify: `src/orca_sim/scenes/v2/scene_right_cube_orientation.xml`
- Create: `tests/test_grasping.py`

**Interfaces:**
- Consumes: existing v2 body names and nominal cube pose.
- Produces: sites `right_palm_grasp_center`, `right_thumb_tip`, `right_index_tip`, `right_middle_tip`, `right_ring_tip`, `right_pinky_tip`; mocap body `task_cube_goal`.

- [ ] **Step 1: Write failing site and goal-body tests**

```python
import numpy as np

from orca_sim import OrcaHandRightCubeOrientation


def test_v2_cube_scene_exposes_grasp_sites() -> None:
    env = OrcaHandRightCubeOrientation(version="v2", goal_mode="cube_orientation")
    expected = (
        "right_palm_grasp_center",
        "right_thumb_tip",
        "right_index_tip",
        "right_middle_tip",
        "right_ring_tip",
        "right_pinky_tip",
    )
    assert all(env.model.site(name).id >= 0 for name in expected)
    palm = env.model.site("right_palm_grasp_center").id
    cube = env.model.body("task_cube").id
    np.testing.assert_allclose(env.data.site_xpos[palm], env.data.xpos[cube], atol=1e-6)


def test_v2_cube_scene_exposes_goal_mocap_body() -> None:
    env = OrcaHandRightCubeOrientation(version="v2", goal_mode="cube_orientation")
    goal = env.model.body("task_cube_goal")
    assert goal.mocapid >= 0
```

- [ ] **Step 2: Run tests and verify missing-name failure**

Run: `uv run pytest tests/test_grasping.py -v`
Expected: FAIL because the named sites and goal body do not exist.

- [ ] **Step 3: Add the model elements**

Inside `right_R-Carpals_8d1f1041`, add the palm site at the nominal cube center expressed in the carpals frame:

```xml
<site name="right_palm_grasp_center" type="sphere"
      pos="0 0.0318770781 0.0281999993" size="0.002"
      rgba="0.2 0.8 1 0"/>
```

Add one zero-offset site inside each distal fingertip body:

```xml
<site name="right_thumb_tip" type="sphere" pos="0 0 0" size="0.0015" rgba="0 0 0 0"/>
<site name="right_index_tip" type="sphere" pos="0 0 0" size="0.0015" rgba="0 0 0 0"/>
<site name="right_middle_tip" type="sphere" pos="0 0 0" size="0.0015" rgba="0 0 0 0"/>
<site name="right_ring_tip" type="sphere" pos="0 0 0" size="0.0015" rgba="0 0 0 0"/>
<site name="right_pinky_tip" type="sphere" pos="0 0 0" size="0.0015" rgba="0 0 0 0"/>
```

Add the non-colliding mocap goal body to the scene:

```xml
<body name="task_cube_goal" mocap="true" pos="0.25 0.12 0.19">
  <geom type="box" size="0.018 0.018 0.018" rgba="0.2 0.8 0.2 0.25"
        contype="0" conaffinity="0" density="0" group="2"/>
</body>
```

- [ ] **Step 4: Run focused and regression tests**

Run: `uv run pytest tests/test_grasping.py tests/test_envs.py tests/test_cube_orientation_goals.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orca_sim/models/v2/mjcf/orcahand_right_body.xml src/orca_sim/scenes/v2/scene_right_cube_orientation.xml tests/test_grasping.py
git commit -m "Add ORCA grasp observation sites"
```

### Task 2: Implement hash-verified grasp artifacts and preparation

**Repository:** `orca_sim`

**Files:**
- Create: `src/orca_sim/grasping.py`
- Modify: `src/orca_sim/__init__.py`
- Extend: `tests/test_grasping.py`

**Interfaces:**
- Produces: `GraspPoseArtifact`, `GraspValidationReport`, `StabilityThresholds`, `PreparationResult`, `write_grasp_artifact`, `load_grasp_artifact`, `write_validation_report`, `load_validation_report`, `smoothstep_targets`, `place_cube_at_palm_center`, `execute_grasp_preparation`.

- [ ] **Step 1: Write failing artifact, trajectory, and placement tests**

```python
import numpy as np
import pytest

from orca_sim.grasping import (
    GraspPoseArtifact,
    load_grasp_artifact,
    smoothstep_targets,
    write_grasp_artifact,
)


def test_grasp_artifact_round_trip_and_tamper_rejection(tmp_path) -> None:
    artifact = GraspPoseArtifact(
        schema_version=1,
        model_version="v2",
        joint_positions={"right_wrist": 0.1, "right_i-mcp": 0.8},
        palm_site_name="right_palm_grasp_center",
        cube_offset_m=(0.0, 0.0, 0.0),
        close_duration_s=0.6,
        hold_duration_s=0.5,
        search_seed=1,
    )
    path = tmp_path / "grasp.json"
    digest = write_grasp_artifact(path, artifact)
    assert load_grasp_artifact(path).content_hash == digest
    path.write_text(path.read_text().replace("0.8", "0.9"))
    with pytest.raises(ValueError, match="content hash"):
        load_grasp_artifact(path)


def test_smoothstep_targets_are_monotonic_and_include_endpoints() -> None:
    target = smoothstep_targets(np.array([0.0]), np.array([1.0]), steps=5)
    np.testing.assert_allclose(target[[0, -1], 0], [0.0, 1.0])
    assert np.all(np.diff(target[:, 0]) >= 0.0)
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `uv run pytest tests/test_grasping.py -v`
Expected: FAIL because `orca_sim.grasping` does not exist.

- [ ] **Step 3: Implement immutable schemas and canonical hashing**

Use these signatures:

```python
@dataclass(frozen=True)
class GraspPoseArtifact:
    schema_version: int
    model_version: str
    joint_positions: dict[str, float]
    palm_site_name: str
    cube_offset_m: tuple[float, float, float]
    close_duration_s: float
    hold_duration_s: float
    search_seed: int
    content_hash: str | None = None


def smoothstep_targets(
    start: np.ndarray, stop: np.ndarray, *, steps: int
) -> np.ndarray:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    u = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    blend = u * u * (3.0 - 2.0 * u)
    return start[None, :] + blend[:, None] * (stop - start)[None, :]


def write_grasp_artifact(path: str | Path, artifact: GraspPoseArtifact) -> str:
    payload = asdict(replace(artifact, content_hash=None))
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    output = {**payload, "content_hash": digest}
    Path(path).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return digest


def load_grasp_artifact(path: str | Path) -> GraspPoseArtifact:
    raw = json.loads(Path(path).read_text())
    stored_hash = raw.pop("content_hash")
    artifact = GraspPoseArtifact(**raw, content_hash=stored_hash)
    expected = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if stored_hash != expected:
        raise ValueError("grasp artifact content hash mismatch")
    return artifact


def place_cube_at_palm_center(
    model,
    data,
    *,
    palm_site_name: str,
    cube_joint_name: str,
    local_offset: np.ndarray,
) -> None:
    mujoco.mj_forward(model, data)
    site_id = model.site(palm_site_name).id
    joint_id = model.joint(cube_joint_name).id
    qpos_adr = int(model.jnt_qposadr[joint_id])
    rotation = data.site_xmat[site_id].reshape(3, 3)
    data.qpos[qpos_adr : qpos_adr + 3] = (
        data.site_xpos[site_id] + rotation @ local_offset
    )
    mujoco.mj_forward(model, data)
```

Canonical hashing uses sorted compact JSON without `content_hash`:

```python
payload = asdict(replace(artifact, content_hash=None))
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
digest = hashlib.sha256(encoded).hexdigest()
```

- [ ] **Step 4: Add stability and preparation data types**

```python
@dataclass(frozen=True)
class StabilityThresholds:
    max_drift_m: float = 0.005
    max_linear_speed_m_s: float = 0.05
    max_angular_speed_rad_s: float = 0.5
    min_contacted_fingers: int = 2


@dataclass(frozen=True)
class GraspValidationReport:
    schema_version: int
    artifact_hash: str
    scenario_mode: str
    episodes: int
    stable_episodes: int
    stability_rate: float
    seed: int
    accepted: bool
    content_hash: str | None = None


@dataclass(frozen=True)
class PreparationResult:
    stable: bool
    final_qpos: tuple[float, ...]
    cube_position: tuple[float, float, float]
    cube_quaternion: tuple[float, float, float, float]
    contacted_fingers: tuple[str, ...]
    max_drift_m: float
    max_linear_speed_m_s: float
    max_angular_speed_rad_s: float
    failure_reasons: tuple[str, ...]


```

Implement `execute_grasp_preparation(env, artifact: GraspPoseArtifact, *, thresholds: StabilityThresholds, seed: int) -> PreparationResult`. The function resets to the open pose, places the cube, sends the 0.6-second smoothstep close trajectory, holds for 0.5 seconds, and evaluates the last 0.25 seconds. It fixes `right_wrist` and rejects non-finite state, cube drop, insufficient contacts, one-sided support, drift, speed, effort, or force violations. The result contains every measured maximum and every failed threshold; it never returns only a Boolean.

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest tests/test_grasping.py -v`
Expected: PASS.

- [ ] **Step 6: Export public artifact types and commit**

```bash
git add src/orca_sim/grasping.py src/orca_sim/__init__.py tests/test_grasping.py
git commit -m "Add stable grasp preparation"
```

### Task 3: Integrate bounded grasp reset into the CPU task

**Repository:** `orca_sim`

**Files:**
- Modify: `src/orca_sim/task_envs.py`
- Create: `tests/test_grasp_reset.py`

**Interfaces:**
- Consumes: `GraspPoseArtifact`, `StabilityThresholds`, `execute_grasp_preparation`.
- Produces: constructor arguments `grasp_artifact`, `grasp_validation_report`, `grasp_reset_attempts`, `grasp_thresholds`; info keys `grasp_stable`, `grasp_artifact_hash`, `grasp_attempts`, `grasp_failure_reasons`.

- [ ] **Step 1: Write failing reset integration tests**

```python
def test_grasp_reset_never_exposes_failed_preparation(monkeypatch, artifact) -> None:
    outcomes = iter([False, False, True])
    monkeypatch.setattr("orca_sim.task_envs.execute_grasp_preparation", lambda *a, **k: fake_result(next(outcomes)))
    env = make_env(grasp_artifact=artifact, grasp_reset_attempts=3)
    _, info = env.reset(seed=7)
    assert info["grasp_stable"] is True
    assert info["grasp_attempts"] == 3


def test_grasp_reset_failure_is_bounded(monkeypatch, artifact) -> None:
    monkeypatch.setattr("orca_sim.task_envs.execute_grasp_preparation", lambda *a, **k: fake_result(False))
    env = make_env(grasp_artifact=artifact, grasp_reset_attempts=2)
    with pytest.raises(RuntimeError, match="stable grasp after 2 attempts"):
        env.reset(seed=7)
```

- [ ] **Step 2: Run and verify constructor failure**

Run: `uv run pytest tests/test_grasp_reset.py -v`
Expected: FAIL because the constructor arguments are unknown.

- [ ] **Step 3: Implement opt-in preparation before task counters start**

Load and hash-check the artifact once in `__init__`. In `reset`, execute bounded preparation attempts before sampling the target. Clear elapsed/task/success counters only after preparation succeeds. Preserve the legacy reset path when no artifact is supplied.

- [ ] **Step 4: Run focused and task regressions**

Run: `uv run pytest tests/test_grasp_reset.py tests/test_cube_orientation_goals.py tests/test_drop_penalty.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orca_sim/task_envs.py tests/test_grasp_reset.py
git commit -m "Add bounded stable grasp resets"
```

### Task 4: Add deterministic CEM grasp search and validation CLI

**Repository:** `orca_sim`

**Files:**
- Create: `src/orca_sim/grasp_search.py`
- Modify: `pyproject.toml`
- Create: `tests/test_grasp_search.py`

**Interfaces:**
- Produces: `CEMSearchConfig`, `CandidateResult`, `SearchResult`, `search_grasp`, `validate_grasp`, CLI `orca-grasp search|validate`.

- [ ] **Step 1: Write failing deterministic sampling and robust-ranking tests**

```python
def test_cem_is_deterministic_for_a_fixed_seed() -> None:
    config = CEMSearchConfig(population=8, elite_count=2, iterations=2, seed=3)
    first = search_grasp(config, evaluator=quadratic_fake_evaluator)
    second = search_grasp(config, evaluator=quadratic_fake_evaluator)
    np.testing.assert_allclose(first.best_joint_positions, second.best_joint_positions)


def test_candidate_ranking_uses_lower_tail_before_mean() -> None:
    robust = CandidateResult(scores=(0.8, 0.8, 0.8), stability_rate=1.0)
    brittle = CandidateResult(scores=(1.0, 1.0, 0.0), stability_rate=2 / 3)
    assert robust.rank_key > brittle.rank_key
```

- [ ] **Step 2: Run and verify import failure**

Run: `uv run pytest tests/test_grasp_search.py -v`
Expected: FAIL because `orca_sim.grasp_search` does not exist.

- [ ] **Step 3: Implement bounded CEM and report generation**

```python
@dataclass(frozen=True)
class CEMSearchConfig:
    population: int = 128
    elite_count: int = 16
    iterations: int = 8
    search_scenarios: int = 32
    validation_episodes: int = 100
    required_stability: float = 0.95
    seed: int = 1


@dataclass(frozen=True)
class CandidateResult:
    scores: tuple[float, ...]
    stability_rate: float
    joint_positions: tuple[float, ...] = ()

    @property
    def rank_key(self) -> tuple[float, float, float]:
        values = np.asarray(self.scores, dtype=np.float64)
        return (
            self.stability_rate,
            float(np.quantile(values, 0.10)),
            float(np.mean(values)),
        )


@dataclass(frozen=True)
class SearchResult:
    best_joint_positions: tuple[float, ...]
    best_candidate: CandidateResult
    artifact_path: str | None
    report_path: str
```

Implement `search_grasp(config: CEMSearchConfig, *, evaluator: Callable[[np.ndarray, int], CandidateResult] | None = None) -> SearchResult` and `validate_grasp(artifact: GraspPoseArtifact, *, episodes: int, seed: int, perturbed: bool) -> GraspValidationReport`. Sample only the 16 finger joints inside ROM margins, keep `right_wrist` fixed, update mean/std from robust elites, and use disjoint search and validation seeds. Exit nonzero and do not write an accepted artifact when validation is below threshold unless `--allow-below-threshold` is explicitly diagnostic.

- [ ] **Step 4: Register and test the CLI**

Add:

```toml
[project.scripts]
orca-grasp = "orca_sim.grasp_search:main"
```

Run: `uv run orca-grasp --help`
Expected: lists `search` and `validate`.

- [ ] **Step 5: Run tests and a tiny physics smoke**

Run: `uv run pytest tests/test_grasp_search.py tests/test_grasping.py tests/test_grasp_reset.py -v`
Run: `uv run orca-grasp search --population 2 --elite-count 1 --iterations 1 --search-scenarios 1 --validation-episodes 1 --allow-below-threshold --output /private/tmp/orca-grasp-smoke.json --report /private/tmp/orca-grasp-smoke-report.json`
Expected: tests pass and diagnostic artifacts are written.

- [ ] **Step 6: Commit**

```bash
git add src/orca_sim/grasp_search.py pyproject.toml tests/test_grasp_search.py
git commit -m "Add robust ORCA grasp search"
```

### Task 5: Add the pinned Playground dependency boundary

**Repository:** `orca_train`

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/orca_train/playground_constants.py`
- Create: `tests/test_playground_constants.py`

**Interfaces:**
- Produces: `WRIST_JOINT`, `ACTIVE_JOINTS`, `ACTIVE_ACTUATORS`, `PALM_SITE`, `FINGERTIP_SITES`, `ACTION_SCHEMA_VERSION`, `OBSERVATION_SCHEMA_VERSION`, `require_playground`.

- [ ] **Step 1: Write failing constant and lazy-dependency tests**

```python
from orca_train.playground_constants import ACTIVE_JOINTS, FINGERTIP_SITES, WRIST_JOINT


def test_orca_playground_mapping_is_16_fingers_plus_fixed_wrist() -> None:
    assert WRIST_JOINT == "right_wrist"
    assert len(ACTIVE_JOINTS) == 16
    assert WRIST_JOINT not in ACTIVE_JOINTS
    assert len(FINGERTIP_SITES) == 5
```

- [ ] **Step 2: Run and verify module import failure**

Run: `uv run pytest tests/test_playground_constants.py -v`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Add the optional dependency and focused constants**

```toml
[project.optional-dependencies]
dev = ["pytest>=8"]
playground = [
  "playground @ git+https://github.com/google-deepmind/mujoco_playground.git@v0.2.0",
]

[tool.hatch.metadata]
allow-direct-references = true
```

`require_playground()` imports JAX, Brax, `mujoco.mjx`, and `mujoco_playground`, verifies the package version/source manifest, and raises an actionable message containing `uv sync --extra playground --group dev` when missing.

- [ ] **Step 4: Resolve and verify dependencies**

Run: `uv sync --extra playground --group dev`
Run: `uv run python -c "import jax, mujoco_playground; from mujoco import mjx; print(jax.default_backend())"`
Expected locally: imports succeed and backend is `cpu` or `metal`; GPU is required only on Spark.

- [ ] **Step 5: Run legacy and new tests**

Run: `uv run pytest tests/test_playground_constants.py tests/test_state_env.py tests/test_train_state.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/orca_train/playground_constants.py tests/test_playground_constants.py
git commit -m "Add pinned Playground integration"
```

### Task 6: Add exact Leap environment and PPO configuration parity

**Repository:** `orca_train`

**Files:**
- Create: `src/orca_train/playground_config.py`
- Create: `tests/test_playground_config.py`

**Interfaces:**
- Produces: `orca_env_config()`, `orca_ppo_config()`, `assert_leap_ppo_parity(config)`, `config_fingerprint(config) -> str`.

- [ ] **Step 1: Write failing field-by-field parity tests**

```python
from mujoco_playground.config import manipulation_params

from orca_train.playground_config import orca_env_config, orca_ppo_config


def test_orca_ppo_config_equals_upstream_leap_config() -> None:
    expected = manipulation_params.brax_ppo_config("LeapCubeReorient", "warp")
    actual = orca_ppo_config()
    assert actual.to_dict() == expected.to_dict()


def test_orca_env_config_copies_leap_values() -> None:
    config = orca_env_config()
    assert (config.ctrl_dt, config.sim_dt, config.action_scale) == (0.05, 0.01, 0.5)
    assert config.episode_length == 1000
    assert config.obs_noise.scales.to_dict() == {"joint_pos": 0.05, "cube_pos": 0.02, "cube_ori": 0.1}
    assert config.reward_config.scales.to_dict() == {
        "orientation": 5.0,
        "position": 0.5,
        "termination": -100.0,
        "hand_pose": -0.5,
        "action_rate": -0.001,
        "joint_vel": 0.0,
        "energy": -0.001,
    }
```

- [ ] **Step 2: Run and verify import failure**

Run: `uv run pytest tests/test_playground_config.py -v`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement config construction from the pinned upstream**

```python
def orca_ppo_config() -> config_dict.ConfigDict:
    config = manipulation_params.brax_ppo_config("LeapCubeReorient", "warp")
    assert_leap_ppo_parity(config)
    return config.copy_and_resolve_references()
```

Construct `orca_env_config` with the exact values in the spec rather than importing the Leap model. Freeze a canonical JSON SHA-256 fingerprint in run manifests.

- [ ] **Step 4: Run parity tests and commit**

Run: `uv run pytest tests/test_playground_config.py -v`
Expected: PASS.

```bash
git add src/orca_train/playground_config.py tests/test_playground_config.py
git commit -m "Add exact Leap PPO configuration"
```

### Task 7: Implement the ORCA Warp/MJX environment core

**Repository:** `orca_train`

**Files:**
- Create: `src/orca_train/playground_env.py`
- Create: `tests/test_playground_env.py`

**Interfaces:**
- Consumes: model constants, environment config, accepted grasp artifact/report.
- Produces: `OrcaCubeReorient`, `uniform_quat`, `load_validated_grasp`, properties `action_size == 16`, `mj_model`, `mjx_model`, `dt`.

- [ ] **Step 1: Write failing model/action/reset tests**

```python
def test_environment_has_16_actions_and_fixed_wrist(validated_grasp) -> None:
    env = OrcaCubeReorient(grasp_artifact=validated_grasp.artifact, validation_report=validated_grasp.report)
    assert env.action_size == 16
    state = env.reset(jax.random.PRNGKey(0))
    wrist_before = state.data.ctrl[env.wrist_actuator_id]
    state = env.step(state, jnp.ones(16))
    assert state.data.ctrl[env.wrist_actuator_id] == wrist_before


def test_reset_shapes_and_zero_velocities(validated_grasp) -> None:
    env = make_env(validated_grasp)
    state = env.reset(jax.random.PRNGKey(1))
    assert state.obs["state"].shape == (57,)
    assert state.obs["privileged_state"].shape == (131,)
    np.testing.assert_allclose(np.asarray(state.data.qvel), 0.0)
```

- [ ] **Step 2: Run and verify module import failure**

Run: `uv run pytest tests/test_playground_env.py -v`
Expected: FAIL because `playground_env` does not exist.

- [ ] **Step 3: Implement model loading and active mapping**

Subclass `mujoco_playground._src.mjx_env.MjxEnv`. Load the absolute ORCA v2 scene path, set `model.opt.timestep=0.01`, verify every required name, patch the validated grasp into the environment's home arrays, and call `mjx.put_model(self._mj_model, impl=self._config.impl)`. Override `action_size` to return 16 even though MuJoCo has 17 actuators.

- [ ] **Step 4: Implement reset exactly around the accepted grasp**

```python
q_finger = jp.clip(grasp_qpos + 0.1 * jax.random.normal(hand_key, (16,)), lower, upper)
cube_pos = grasp_cube_pos + jax.random.uniform(pos_key, (3,), minval=-0.01, maxval=0.01)
cube_quat = uniform_quat(quat_key)
qvel = jp.zeros(mj_model.nv)
ctrl = full_ctrl.at[active_actuator_ids].set(q_finger)
ctrl = ctrl.at[wrist_actuator_id].set(fixed_wrist)
```

Sample the reset goal with a separate uniform quaternion. Initialize histories and previous actions to zeros. Terminate on non-finite state or cube height below `nominal_grasp_z - 0.10`.

- [ ] **Step 5: Implement relative actions and physics step**

```python
delta = jp.clip(action, -1.0, 1.0) * config.action_scale
finger_targets = jp.clip(state.data.ctrl[active_ids] + delta, lower, upper)
motor_targets = state.data.ctrl.at[active_ids].set(finger_targets)
motor_targets = motor_targets.at[wrist_id].set(fixed_wrist)
data = mjx_env.step(mjx_model, state.data, motor_targets, n_substeps)
```

- [ ] **Step 6: Run CPU JAX and scalar rollout tests**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_playground_env.py -v`
Expected: PASS with finite reset and step state.

- [ ] **Step 7: Commit**

```bash
git add src/orca_train/playground_env.py tests/test_playground_env.py
git commit -m "Add ORCA Playground environment"
```

### Task 8: Implement exact 57/131 observations and Leap reward

**Repository:** `orca_train`

**Files:**
- Modify: `src/orca_train/playground_env.py`
- Extend: `tests/test_playground_env.py`

**Interfaces:**
- Produces: `_get_obs(data, info)`, `_get_reward(data, action, info, done)`, `_cube_orientation_error(data)`.

- [ ] **Step 1: Write failing slice and noise-free observation tests**

```python
def test_actor_and_critic_field_slices(no_noise_env, reset_state) -> None:
    obs = no_noise_env._get_obs(reset_state.data, reset_state.info)
    assert obs["state"].shape == (57,)
    assert obs["privileged_state"].shape == (131,)
    np.testing.assert_allclose(obs["state"][:16], reset_state.data.qpos[no_noise_env.hand_qids])
    np.testing.assert_allclose(obs["state"][41:57], reset_state.info["last_act"])
    np.testing.assert_allclose(obs["privileged_state"][89:104], no_noise_env.fingertip_positions(reset_state.data))
```

- [ ] **Step 2: Write failing reward reference tests**

Construct fixed JAX arrays and compare orientation tolerance, position tolerance, hand-pose cost, first/second action-rate cost, energy cost, timestep multiplication, and `+100` success bonus against NumPy reference calculations.

- [ ] **Step 3: Run and verify shape/slice/reward failures**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_playground_env.py -v`
Expected: FAIL at the first unimplemented observation or reward assertion.

- [ ] **Step 4: Implement observations in this exact order**

```python
state = jp.concatenate([
    noisy_joint_angles,                 # 0:16
    qpos_error_history,                 # 16:32
    cube_pos_error_history,             # 32:35
    cube_ori_error_history,             # 35:41
    info["last_act"],                   # 41:57
])
privileged_state = jp.concatenate([
    state,
    clean_joint_pos,                    # 57:73
    clean_joint_vel,                    # 73:89
    fingertip_positions_relative_palm,  # 89:104
    clean_cube_pos_error,               # 104:107
    clean_cube_ori_error,               # 107:113
    cube_linvel,                         # 113:116
    cube_angvel,                         # 116:119
    info["pert_dir"],                   # 119:125
    cube_wrench,                         # 125:131
])
```

Use the exact Leap noise distributions and relative rotation matrix `quat_to_mat(cube_quat * inv(goal_quat)).ravel()[3:]`.

- [ ] **Step 5: Implement reward and moving goal update**

Copy the pinned Leap formulas and scales without extra terms. In the full-SO(3) stage, copy the success-triggered `goal_quat_dquat` integration and 0.8 decay exactly.

- [ ] **Step 6: Run tests and commit**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_playground_env.py -v`
Expected: PASS.

```bash
git add src/orca_train/playground_env.py tests/test_playground_env.py
git commit -m "Add asymmetric Leap observations and reward"
```

### Task 9: Implement semantic domain randomization

**Repository:** `orca_train`

**Files:**
- Modify: `src/orca_train/playground_env.py`
- Create: `tests/test_playground_randomization.py`

**Interfaces:**
- Produces: `domain_randomize(model: mjx.Model, rng: jax.Array)` returning `(model, in_axes)` for Brax wrappers.

- [ ] **Step 1: Write failing bounds and field-selection tests**

```python
def test_domain_randomization_preserves_fixed_wrist_and_changes_fingers(env) -> None:
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    randomized, _ = domain_randomize(env.mjx_model, keys)
    wrist = env.wrist_qpos_id
    np.testing.assert_allclose(np.asarray(randomized.qpos0[:, wrist]), env.mjx_model.qpos0[wrist])
    assert np.ptp(np.asarray(randomized.qpos0[:, env.hand_qids[0]])) > 0.0


def test_five_fingertip_frictions_stay_in_leap_bounds(env) -> None:
    randomized, _ = domain_randomize(env.mjx_model, jax.random.split(jax.random.PRNGKey(1), 64))
    values = np.asarray(randomized.geom_friction[:, env.fingertip_geom_ids, 0])
    assert values.min() >= 0.5
    assert values.max() <= 1.0
```

- [ ] **Step 2: Run and verify missing-function failure**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_playground_randomization.py -v`
Expected: FAIL because `domain_randomize` does not exist.

- [ ] **Step 3: Port the upstream randomizer with ORCA IDs**

Apply the exact intervals from the spec to five fingertip collision geoms, cube inertia and inertial position, the 16 active qpos entries, finger friction loss, armature, hand link masses, active actuator gains/biases, and active joint damping. Return a correct batched `in_axes`; never randomize the wrist policy mapping.

- [ ] **Step 4: Run tests and commit**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_playground_randomization.py tests/test_playground_env.py -v`
Expected: PASS.

```bash
git add src/orca_train/playground_env.py tests/test_playground_randomization.py
git commit -m "Add ORCA Leap domain randomization"
```

### Task 10: Add the six-stage checkpointed curriculum

**Repository:** `orca_train`

**Files:**
- Create: `src/orca_train/playground_curriculum.py`
- Modify: `src/orca_train/playground_env.py`
- Create: `tests/test_playground_curriculum.py`

**Interfaces:**
- Produces: `CurriculumStage`, `CurriculumState`, `ORCA_REORIENT_CURRICULUM`, `sample_goal`, `update_curriculum`.

- [ ] **Step 1: Write failing stage and promotion tests**

```python
def test_curriculum_has_hold_to_full_so3_stages() -> None:
    assert [stage.name for stage in ORCA_REORIENT_CURRICULUM] == [
        "hold", "turn_10_15", "turn_30", "turn_60", "turn_90", "full_so3"
    ]


def test_promotion_requires_two_consecutive_passing_evaluations() -> None:
    state = CurriculumState(stage_index=0)
    state = update_curriculum(state, {"success_rate": 0.99, "drop_rate": 0.01})
    assert (state.stage_index, state.promotion_streak) == (0, 1)
    state = update_curriculum(state, {"success_rate": 1.0, "drop_rate": 0.0})
    assert (state.stage_index, state.promotion_streak) == (1, 0)
```

- [ ] **Step 2: Run and verify import failure**

Run: `uv run pytest tests/test_playground_curriculum.py -v`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement immutable stage definitions**

```python
ORCA_REORIENT_CURRICULUM = (
    CurriculumStage("hold", (0.0, 0.0), 0.99, 0.01),
    CurriculumStage("turn_10_15", (np.deg2rad(10), np.deg2rad(15)), 0.80, 0.05),
    CurriculumStage("turn_30", (np.deg2rad(30), np.deg2rad(30)), 0.80, 0.05),
    CurriculumStage("turn_60", (np.deg2rad(60), np.deg2rad(60)), 0.80, 0.05),
    CurriculumStage("turn_90", (np.pi / 2, np.pi / 2), 0.80, 0.05),
    CurriculumStage("full_so3", None, 0.80, 0.05),
)
```

For bounded stages, sample a uniform axis and an angle in the stage interval relative to the current cube orientation. For `full_so3`, use the exact Leap reset and moving-goal behavior. Validate finite rates in `[0, 1]`; reset streak on failure; never advance beyond stage 5.

- [ ] **Step 4: Integrate stage selection in reset/step and run tests**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_playground_curriculum.py tests/test_playground_env.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orca_train/playground_curriculum.py src/orca_train/playground_env.py tests/test_playground_curriculum.py
git commit -m "Add full orientation curriculum"
```

### Task 11: Add manifests, Brax PPO training, and checkpoint resume

**Repository:** `orca_train`

**Files:**
- Create: `src/orca_train/playground_run.py`
- Create: `tests/test_playground_run.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `RunManifest`, `classify_run`, `build_train_fn`, `train_playground`, CLI `orca-train-playground`.

- [ ] **Step 1: Write failing manifest and diagnostic-label tests**

```python
def test_reduced_config_is_always_diagnostic(validated_grasp) -> None:
    manifest = RunManifest.create(ppo_config=replace_timesteps(1_000), grasp=validated_grasp)
    assert manifest.classification == "diagnostic"


def test_exact_config_can_be_final_candidate(validated_grasp) -> None:
    manifest = RunManifest.create(ppo_config=orca_ppo_config(), grasp=validated_grasp)
    assert manifest.classification == "final_candidate"


def test_resume_rejects_observation_or_grasp_mismatch(tmp_path) -> None:
    with pytest.raises(ValueError, match="grasp_artifact_hash"):
        validate_resume(tmp_path / "checkpoint", current_manifest=different_grasp_manifest())
```

- [ ] **Step 2: Run and verify import failure**

Run: `uv run pytest tests/test_playground_run.py -v`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the run manifest and JSONL progress callback**

Use this immutable manifest boundary:

```python
@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    classification: str
    seed: int
    git_revisions: dict[str, str]
    dependency_lock_sha256: str
    playground_revision: str
    environment_config: dict[str, object]
    ppo_config: dict[str, object]
    action_schema_version: int
    observation_schema_version: int
    active_joint_names: tuple[str, ...]
    fixed_wrist_position: float
    grasp_artifact_hash: str
    grasp_validation_hash: str
    curriculum_stage: int
    curriculum_promotion_streak: int
    backend: str
```

`RunManifest.create` assigns `final_candidate` only when the config fingerprint equals the exact upstream parity fingerprint; every override produces `diagnostic`. Record all revisions, dependency lock SHA-256, effective configs, schemas, joint order, wrist target, grasp/report hashes, curriculum state, seed, and backend. Write manifests atomically and append progress metrics as one JSON object per line.

- [ ] **Step 4: Build PPO directly from the pinned runner API**

```python
training_params = dict(orca_ppo_config())
network_params = training_params.pop("network_factory")
num_eval_envs = training_params.pop("num_eval_envs", 128)
network_factory = functools.partial(ppo_networks.make_ppo_networks, **network_params)
train_fn = functools.partial(
    ppo.train,
    **training_params,
    network_factory=network_factory,
    seed=seed,
    restore_checkpoint_path=resume,
    save_checkpoint_path=checkpoint_dir,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    num_eval_envs=num_eval_envs,
)
```

Enable `domain_randomize` for the final run. For diagnostic flags, permit only explicit `--diagnostic-num-timesteps` and `--diagnostic-num-envs`; classification must remain diagnostic.

- [ ] **Step 5: Add CLI and a one-update smoke test**

Add:

```toml
orca-train-playground = "orca_train.playground_run:main"
```

Run: `JAX_PLATFORMS=cpu uv run orca-train-playground --grasp-artifact tests/fixtures/grasp.json --validation-report tests/fixtures/grasp-validation.json --diagnostic-num-timesteps 1024 --diagnostic-num-envs 8 --output-dir /private/tmp/orca-playground-smoke`
Expected: finite metrics and a restorable diagnostic checkpoint.

- [ ] **Step 6: Run tests and commit**

Run: `uv run pytest tests/test_playground_run.py tests/test_playground_config.py tests/test_playground_curriculum.py -v`
Expected: PASS.

```bash
git add pyproject.toml src/orca_train/playground_run.py tests/test_playground_run.py
git commit -m "Add ORCA Brax PPO runner"
```

### Task 12: Add held-out evaluation, export, and video

**Repository:** `orca_train`

**Files:**
- Create: `src/orca_train/playground_evaluate.py`
- Create: `src/orca_train/playground_export.py`
- Create: `src/orca_train/playground_video.py`
- Create: `tests/test_playground_export.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `EvaluationReport`, `evaluate_checkpoint`, `accept_checkpoint`, `export_policy_bundle`, `verify_export`, CLIs `orca-evaluate-playground`, `orca-export-playground`, `orca-render-playground`.

- [ ] **Step 1: Write failing acceptance and export-parity tests**

```python
def test_acceptance_requires_final_so3_exact_run() -> None:
    report = fake_report(stage="turn_90", success_rate=1.0, drop_rate=0.0)
    assert accept_checkpoint(report) is False


def test_acceptance_requires_80_success_and_5_drop() -> None:
    assert accept_checkpoint(fake_report(stage="full_so3", success_rate=0.80, drop_rate=0.05))
    assert not accept_checkpoint(fake_report(stage="full_so3", success_rate=0.79, drop_rate=0.0))
    assert not accept_checkpoint(fake_report(stage="full_so3", success_rate=1.0, drop_rate=0.06))


def test_exported_policy_matches_checkpoint_actions(tmp_path, trained_fixture) -> None:
    bundle = export_policy_bundle(trained_fixture.checkpoint, tmp_path / "policy.pkl")
    assert verify_export(bundle, trained_fixture.fixed_observation_batch, atol=1e-6)
```

- [ ] **Step 2: Run and verify import failure**

Run: `uv run pytest tests/test_playground_export.py -v`
Expected: FAIL because evaluation/export modules do not exist.

- [ ] **Step 3: Implement fixed-seed evaluation and acceptance reports**

Use this report boundary:

```python
@dataclass(frozen=True)
class EvaluationReport:
    schema_version: int
    run_classification: str
    curriculum_stage: str
    episodes: int
    success_rate: float
    drop_rate: float
    mean_goals_completed: float
    mean_orientation_error_rad: float
    mean_episode_reward: float
    max_wrist_drift_rad: float
    nonfinite_episodes: int
    checkpoint_sha256: str
    manifest_sha256: str
    accepted: bool
    per_seed: tuple[dict[str, object], ...]
```

Evaluate stage 5 over at least 100 held-out reset seeds with domain randomization. Report success/drop rates, goals completed, orientation error, episode reward, wrist drift, NaNs, per-seed outcomes, checkpoint hash, and manifest hash. `accept_checkpoint` returns true only for `final_candidate`, `full_so3`, at least 100 episodes, success at least 0.80, drop at most 0.05, zero non-finite episodes, and wrist drift at most numerical tolerance.

- [ ] **Step 4: Implement policy bundle and deterministic verification**

Bundle policy parameters, observation normalizer state, inference metadata, active joint order, schemas, wrist target, grasp hash, config fingerprints, and source checkpoint hash. Compare deterministic actions from restored checkpoint and bundle on a fixed observation batch before writing `verified=true`.

- [ ] **Step 5: Implement rollout rendering and register CLIs**

Use the accepted policy, collect a JAX rollout, convert states to the environment renderer, and write MP4 at `1 / env.dt / render_every`. Refuse non-accepted checkpoints unless `--allow-diagnostic` is explicit and watermark the output filename as diagnostic.

- [ ] **Step 6: Run tests and commit**

Run: `uv run pytest tests/test_playground_export.py tests/test_playground_run.py -v`
Expected: PASS.

```bash
git add pyproject.toml src/orca_train/playground_evaluate.py src/orca_train/playground_export.py src/orca_train/playground_video.py tests/test_playground_export.py
git commit -m "Add PPO evaluation and export"
```

### Task 13: Run complete local verification and document operations

**Repositories:** `orca_sim`, `orca_train`

**Files:**
- Modify: `README.md` in both repositories as needed.

- [ ] **Step 1: Document exact workflows**

Document these complete commands with required environment variables, artifact paths, resume rules, and diagnostic/final distinctions:

```bash
uv run orca-grasp search --population 128 --elite-count 16 --iterations 8 --search-scenarios 32 --validation-episodes 100 --required-stability 0.95 --seed 1 --output runs/grasp_search_seed1/grasp_pose.json --report runs/grasp_search_seed1/search_report.json
uv run orca-grasp validate --artifact runs/grasp_search_seed1/grasp_pose.json --episodes 100 --seed 20001 --scenario-mode perturbed --required-stability 0.95 --report runs/grasp_search_seed1/validation_report.json
uv run orca-train-playground --grasp-artifact ../orca_sim/runs/grasp_search_seed1/grasp_pose.json --validation-report ../orca_sim/runs/grasp_search_seed1/validation_report.json --seed 1 --domain-randomization --output-dir runs/orca_leap_reorient_seed1
uv run orca-evaluate-playground --checkpoint runs/orca_leap_reorient_seed1/checkpoints --episodes 100 --seed 40001 --domain-randomization --report runs/orca_leap_reorient_seed1/final_evaluation.json
uv run orca-export-playground --checkpoint runs/orca_leap_reorient_seed1/checkpoints --evaluation-report runs/orca_leap_reorient_seed1/final_evaluation.json --output runs/orca_leap_reorient_seed1/policy_bundle.pkl
uv run orca-render-playground --policy runs/orca_leap_reorient_seed1/policy_bundle.pkl --output runs/orca_leap_reorient_seed1/accepted_rollout.mp4 --seed 50001
```

- [ ] **Step 2: Run the complete `orca_sim` suite**

Run: `uv run pytest tests/`
Expected: PASS with no failures.

- [ ] **Step 3: Run the complete `orca_train` suite with Playground**

Run: `uv sync --extra playground --group dev`
Run: `JAX_PLATFORMS=cpu uv run pytest tests/`
Expected: PASS with no failures.

- [ ] **Step 4: Run deterministic physics and PPO smoke commands**

Run the two-candidate grasp smoke and the 8-environment/1024-step PPO smoke from Tasks 4 and 11. Verify manifests, hashes, finite metrics, checkpoint restore, export parity, and diagnostic labels.

- [ ] **Step 5: Review diffs and commit documentation**

Run: `git diff --check` in both repositories.
Expected: no output.

```bash
git add README.md
git commit -m "Document ORCA Leap PPO workflow"
```

### Task 14: Execute Spark preflight and stable-grasp acceptance

**Remote root:** `/home/jerry/Code/orca_leap_ppo_20260820`

- [ ] **Step 1: Create isolated remote directories and sync exact worktrees**

Use `rsync -azR` to copy the two implementation worktrees while excluding `.git`, `.venv`, caches, and local run outputs. Write local/remote Git revisions into `source-revisions.json`.

- [ ] **Step 2: Create the remote `uv` environment and install CUDA JAX**

```bash
python3 -m venv /home/jerry/Code/orca_leap_ppo_20260820/.venv
/home/jerry/Code/orca_leap_ppo_20260820/.venv/bin/pip install uv
cd /home/jerry/Code/orca_leap_ppo_20260820/orca_train
../.venv/bin/uv sync --extra playground --group dev
../.venv/bin/uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple
```

- [ ] **Step 3: Verify the GB10 GPU backend and model compilation**

```bash
unset LD_LIBRARY_PATH
export JAX_DEFAULT_MATMUL_PRECISION=highest
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
uv run pytest tests/test_playground_env.py tests/test_playground_randomization.py -v
```

Expected: backend `gpu`, an NVIDIA GB10 device, and passing Warp/MJX tests.

- [ ] **Step 4: Launch CEM search in a named persistent session**

Run `orca-grasp search` with population 128, elite count 16, 8 iterations, 32 search scenarios, 100 validation episodes, required stability 0.95, seed 1. Preserve stdout/stderr, artifact, and report under `runs/grasp_search_seed1/`.

- [ ] **Step 5: Run independent nominal and perturbed validation**

Run 100 nominal holds with required stability 1.0 and 100 held-out perturbed resets with required stability 0.95 using disjoint seeds. Stop before PPO if either gate fails.

- [ ] **Step 6: Copy accepted grasp artifacts locally and verify hashes**

Copy the artifact, search report, nominal report, perturbed report, and logs to `orca_train/runs/grasp_search_seed1_spark/`. Recompute and compare SHA-256 hashes locally and remotely.

### Task 15: Run exact PPO training, evaluation, and artifact handoff

**Remote root:** `/home/jerry/Code/orca_leap_ppo_20260820`

- [ ] **Step 1: Run the remote diagnostic PPO smoke**

Use 8 then 512 environments and short timesteps. Require GPU backend, finite metrics, fixed wrist, checkpoint restore, curriculum persistence, domain randomization, and policy export parity. Keep both runs classified `diagnostic`.

- [ ] **Step 2: Launch the exact final-candidate run**

```bash
export JAX_DEFAULT_MATMUL_PRECISION=highest
uv run orca-train-playground \
  --grasp-artifact ../orca_sim/runs/grasp_search_seed1/grasp_pose.json \
  --validation-report ../orca_sim/runs/grasp_search_seed1/validation_report.json \
  --seed 1 \
  --domain-randomization \
  --output-dir runs/orca_leap_reorient_seed1
```

Do not pass diagnostic overrides. Verify the manifest says 200,000,000 steps, 8192 envs, and `final_candidate` before leaving the persistent session running.

- [ ] **Step 3: Monitor without changing the approved PPO config**

Capture progress at each of 20 evaluations, curriculum stage transitions, success/drop rates, throughput, memory, JIT time, NaNs, and checkpoint paths. Resume only from manifest-compatible checkpoints after interruption.

- [ ] **Step 4: Evaluate the best full-SO(3) checkpoint**

Run at least 100 held-out seeds with domain randomization. Require full SO(3), success at least 80%, drop at most 5%, no wrist drift, and no non-finite episodes.

- [ ] **Step 5: Export and render only after acceptance**

Create the verified policy-only bundle and representative MP4 from the accepted checkpoint. If no checkpoint passes, preserve reports and mark the run rejected rather than exporting it as final.

- [ ] **Step 6: Copy the complete accepted run locally**

Copy best checkpoint, policy bundle, normalization state, manifests, config snapshots, metrics, evaluation report, curriculum report, grasp artifacts, and MP4 into `orca_train/runs/orca_leap_reorient_seed1_spark/`. Verify hashes after transfer.

- [ ] **Step 7: Run final local artifact verification**

Restore the copied checkpoint, load the policy bundle, compare deterministic actions on the fixed observation batch, inspect the video, and record the exact local paths and final metrics in the handoff.

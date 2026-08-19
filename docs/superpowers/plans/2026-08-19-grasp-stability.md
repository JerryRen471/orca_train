# Grasp Stability and Drop Prevention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a sim-to-real-compatible palm-centered grasp reset, automatically find a robust ORCA Hand grasp pose, and train cube rotation through a stability-first curriculum that cannot profit from dropping the cube.

**Architecture:** `orca_sim` owns the physical reset protocol: palm-site placement, smooth closing, contact/stability qualification, bounded retries, target-angle generation, and robust CEM grasp search. `orca_train` consumes a hash-verified grasp artifact, applies safe residual actions and grasp-aware reward, advances a checkpointed five-stage curriculum, and reports stability metrics. Large grasp search, acceptance evaluation, and RL training run on Spark only.

**Tech Stack:** Python 3.10-3.12, NumPy, MuJoCo, Gymnasium, PyTorch, pytest, uv, JSON artifacts, SSH/rsync for Spark execution.

**Spec:** `docs/superpowers/specs/2026-08-19-grasp-stability-design.md`

## Global Constraints

- Start every rotation episode from a physically simulated stable grasp.
- Initialize the cube at the named center of the palm in the hand frame, under normal gravity and contact physics.
- Do not use a weld, mocap attachment, invisible support geometry, gravity change, or simulation-only retention mechanism.
- A human or fixture may place the cube at the palm center; autonomous pickup is out of scope.
- The closing trajectory must obey the model joint ROM and be executable by the physical ORCA Hand controller.
- Contact data may qualify reset and shape training reward, but must not be added to the deployed policy observation.
- Default preparation timing is 0.6 seconds closing, 0.5 seconds holding, with the final 0.25 seconds used for stability qualification.
- Default stability limits are 5 mm palm-relative drift, 0.05 m/s relative linear speed, and 0.5 rad/s relative angular speed, with palm contact and at least two distinct finger contacts.
- The selected grasp must achieve at least 95% stability on held-out perturbations.
- Stage 0 requires at least 99% hold success and at most 1% drops; rotation stages require at least 80% success and at most 5% drops.
- Full CEM search and RL training run on Spark. Local runs are limited to unit tests and short deterministic physics checks.
- Use `uv` for all Python and dependency commands in both repositories.
- Preserve existing state-policy observation shape `(47,)` and normalized action contract; checkpoint incompatibilities must fail explicitly.

## File Structure

### `orca_sim`

- Modify `src/orca_sim/scenes/v2/scene_right_cube_orientation.xml` — declare the named palm-center site used by reset placement.
- Create `src/orca_sim/grasping.py` — grasp artifact schema/hash, palm placement, close trajectory, contact summary, stability samples, preparation controller, and structured failures.
- Create `src/orca_sim/grasp_search.py` — deterministic CEM, perturbation evaluation, artifact/report output, and `search`/`validate` CLI.
- Modify `src/orca_sim/task_envs.py` — opt-in grasp preparation, curriculum-sized hand-X targets, task-start reset, and grasp diagnostic info.
- Modify `src/orca_sim/__init__.py` — export public grasp configuration/artifact types.
- Modify `pyproject.toml` — register `orca-grasp` CLI.
- Create `tests/test_grasping.py` — placement, artifacts, trajectory, contact classification, stability, and failure tests.
- Create `tests/test_grasp_reset.py` — environment integration and task-clock tests.
- Create `tests/test_grasp_search.py` — deterministic CEM, robust ranking, report, and CLI tests.
- Modify `tests/test_cube_orientation_goals.py` — curriculum target-angle and orientation-delta info tests.

### `orca_train`

- Create `src/orca_train/grasp_reward.py` — grasp-quality gate, reward breakdown, drop-bound validation, and action regularization.
- Create `src/orca_train/curriculum.py` — immutable five-stage definitions and deterministic promotion state machine.
- Modify `src/orca_train/state_env.py` — artifact-backed safety envelope, policy takeover continuity, reward replacement, and action/grasp metrics.
- Modify `src/orca_train/state_agent.py` — checkpoint run metadata and compatibility validation.
- Modify `src/orca_train/train_state.py` — grasp curriculum preset/config/CLI, safe replay seeding, episode-boundary stage promotion, environment rebuild, checkpoint state, and richer evaluation.
- Modify `pyproject.toml` — no new dependency; existing `orca-train-state` entry point remains the training CLI.
- Create `tests/test_grasp_reward.py` — falling exploit, drop dominance, and action penalty tests.
- Create `tests/test_curriculum.py` — stage definitions, promotion streak, terminal-stage, and serialization tests.
- Modify `tests/test_state_env.py` — safety envelope, takeover continuity, and reward/metric tests.
- Modify `tests/test_state_agent.py` — run-metadata round trip and mismatch tests.
- Modify `tests/test_train_state.py` — preset/CLI, safe replay seeding, evaluation metrics, promotion, resume, and environment rebuild tests.
- Modify `README.md` — exact grasp search, validation, curriculum training, and resume commands.

---

### Task 1: Add palm-centered placement and a hash-verified grasp artifact

**Repository:** `orca_sim`

**Files:**
- Modify: `src/orca_sim/scenes/v2/scene_right_cube_orientation.xml:14-21`
- Create: `src/orca_sim/grasping.py`
- Modify: `src/orca_sim/__init__.py:1-24`
- Test: `tests/test_grasping.py`

**Interfaces:**
- Consumes: MuJoCo `MjModel`/`MjData`, cube free-joint qpos address, and the v2 scene.
- Produces: `GraspPoseArtifact`, `load_grasp_artifact(path)`, `write_grasp_artifact(path, artifact)`, `place_cube_at_palm_center(...)`, and named site `right_palm_grasp_center`.

- [ ] **Step 1: Write failing site-placement and artifact tests**

Create `tests/test_grasping.py` with these initial tests:

```python
import json

import mujoco
import numpy as np
import pytest

from orca_sim import OrcaHandRightCubeOrientation
from orca_sim.grasping import (
    GraspPoseArtifact,
    load_grasp_artifact,
    place_cube_at_palm_center,
    write_grasp_artifact,
)


def test_v2_scene_exposes_palm_center_site() -> None:
    env = OrcaHandRightCubeOrientation(version="v2")
    try:
        site_id = env.model.site("right_palm_grasp_center").id
        np.testing.assert_allclose(
            env.data.site_xpos[site_id],
            np.array([0.17, -0.015, 0.19]),
            atol=1e-6,
        )
    finally:
        env.close()


def test_cube_is_placed_at_palm_site_plus_local_offset() -> None:
    env = OrcaHandRightCubeOrientation(version="v2")
    try:
        mujoco.mj_resetData(env.model, env.data)
        mujoco.mj_forward(env.model, env.data)
        place_cube_at_palm_center(
            env.model,
            env.data,
            cube_qpos_adr=env._cube_qpos_adr,
            palm_site_name="right_palm_grasp_center",
            local_offset_m=np.array([0.001, -0.002, 0.003]),
        )
        site = env.model.site("right_palm_grasp_center").id
        rotation = env.data.site_xmat[site].reshape(3, 3)
        expected = env.data.site_xpos[site] + rotation @ np.array(
            [0.001, -0.002, 0.003]
        )
        np.testing.assert_allclose(
            env.data.qpos[env._cube_qpos_adr : env._cube_qpos_adr + 3], expected
        )
    finally:
        env.close()


def test_grasp_artifact_hash_round_trip_and_tamper_rejection(tmp_path) -> None:
    path = tmp_path / "grasp_pose.json"
    artifact = GraspPoseArtifact(
        schema_version=1,
        joint_positions={"right_i-mcp": 0.7, "right_t-cmc": -0.2},
        close_duration_s=0.6,
        hold_duration_s=0.5,
        palm_site_name="right_palm_grasp_center",
        palm_offset_m=(0.0, 0.0, 0.0),
        validation_episodes=100,
        validation_stability_rate=0.97,
    )
    content_hash = write_grasp_artifact(path, artifact)
    loaded = load_grasp_artifact(path)
    assert loaded == artifact
    assert loaded.content_hash == content_hash

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["joint_positions"]["right_i-mcp"] = 0.8
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash"):
        load_grasp_artifact(path)
```

- [ ] **Step 2: Run the focused tests and verify the missing interfaces fail**

Run:

```bash
cd /Users/jerry/Code/orca_sim
uv run pytest tests/test_grasping.py -v
```

Expected: collection fails because `orca_sim.grasping` does not exist, and the site lookup would fail before the XML change.

- [ ] **Step 3: Add the named site and artifact/placement implementation**

Inside `right_mount` in `scene_right_cube_orientation.xml`, before the hand-body include, add:

```xml
<site
  name="right_palm_grasp_center"
  type="sphere"
  pos="-0.015 0.17 -0.03"
  size="0.002"
  rgba="0.2 0.8 1 0"
/>
```

Implement `grasping.py` with immutable validation and canonical SHA-256 hashing. The hash covers the sorted JSON payload but not the `content_hash` field itself:

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class GraspPoseArtifact:
    schema_version: int
    joint_positions: dict[str, float]
    close_duration_s: float
    hold_duration_s: float
    palm_site_name: str
    palm_offset_m: tuple[float, float, float]
    validation_episodes: int
    validation_stability_rate: float

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("grasp artifact schema_version must be 1")
        if not self.joint_positions:
            raise ValueError("grasp artifact must contain joint positions")
        values = np.asarray(list(self.joint_positions.values()), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("grasp joint positions must be finite")
        if self.close_duration_s <= 0.0 or self.hold_duration_s <= 0.0:
            raise ValueError("grasp durations must be positive")
        if len(self.palm_offset_m) != 3 or not np.all(
            np.isfinite(self.palm_offset_m)
        ):
            raise ValueError("palm_offset_m must contain three finite values")
        if self.validation_episodes <= 0:
            raise ValueError("validation_episodes must be positive")
        if not 0.0 <= self.validation_stability_rate <= 1.0:
            raise ValueError("validation_stability_rate must be in [0, 1]")

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(_canonical_payload(self)).hexdigest()


def _artifact_payload(artifact: GraspPoseArtifact) -> dict[str, Any]:
    payload = asdict(artifact)
    payload["joint_positions"] = dict(sorted(artifact.joint_positions.items()))
    payload["palm_offset_m"] = list(artifact.palm_offset_m)
    return payload


def _canonical_payload(artifact: GraspPoseArtifact) -> bytes:
    return json.dumps(
        _artifact_payload(artifact),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def write_grasp_artifact(path: str | Path, artifact: GraspPoseArtifact) -> str:
    destination = Path(path)
    payload = _artifact_payload(artifact)
    payload["content_hash"] = artifact.content_hash
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return artifact.content_hash


def load_grasp_artifact(path: str | Path) -> GraspPoseArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_hash = payload.pop("content_hash", None)
    payload["palm_offset_m"] = tuple(payload["palm_offset_m"])
    artifact = GraspPoseArtifact(**payload)
    if expected_hash != artifact.content_hash:
        raise ValueError("grasp artifact content hash mismatch")
    return artifact


def place_cube_at_palm_center(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    cube_qpos_adr: int,
    palm_site_name: str,
    local_offset_m: np.ndarray,
) -> np.ndarray:
    offset = np.asarray(local_offset_m, dtype=np.float64)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("local_offset_m must have shape (3,) and be finite")
    site_id = model.site(palm_site_name).id
    site_rotation = data.site_xmat[site_id].reshape(3, 3)
    cube_position = data.site_xpos[site_id] + site_rotation @ offset
    data.qpos[cube_qpos_adr : cube_qpos_adr + 3] = cube_position
    return cube_position.copy()
```

Export `GraspPoseArtifact`, `load_grasp_artifact`, and `write_grasp_artifact` from `orca_sim.__init__`.

- [ ] **Step 4: Run focused tests and the existing scene smoke test**

Run:

```bash
cd /Users/jerry/Code/orca_sim
uv run pytest tests/test_grasping.py tests/test_envs.py::test_requested_control_period_resolves_to_integral_simulator_steps -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
cd /Users/jerry/Code/orca_sim
git add src/orca_sim/scenes/v2/scene_right_cube_orientation.xml src/orca_sim/grasping.py src/orca_sim/__init__.py tests/test_grasping.py
git commit -m "Add palm-centered grasp artifacts"
```

---

### Task 2: Implement transferable closing trajectories and the stability gate

**Repository:** `orca_sim`

**Files:**
- Modify: `src/orca_sim/grasping.py`
- Modify: `tests/test_grasping.py`

**Interfaces:**
- Consumes: `GraspPoseArtifact`, MuJoCo contact data, actuator qpos indices, and configured palm/finger root bodies.
- Produces: `GraspGeometryConfig`, `GraspStabilityThresholds`, `GraspContactSummary`, `GraspStabilitySample`, `GraspStabilityResult`, and these exact functions:

```python
def smooth_close_targets(
    opened: np.ndarray,
    grasp: np.ndarray,
    *,
    steps: int,
    joint_low: np.ndarray,
    joint_high: np.ndarray,
) -> np.ndarray: ...

def summarize_cube_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    cube_body_id: int,
    geometry: GraspGeometryConfig,
) -> GraspContactSummary: ...

def evaluate_grasp_stability(
    samples: Sequence[GraspStabilitySample],
    thresholds: GraspStabilityThresholds,
) -> GraspStabilityResult: ...
```

- [ ] **Step 1: Add failing trajectory, contact, and stability tests**

Append tests that assert the exact behavior:

```python
from orca_sim.grasping import (
    GraspContactSummary,
    GraspStabilitySample,
    GraspStabilityThresholds,
    evaluate_grasp_stability,
    smooth_close_targets,
)


def test_smooth_close_targets_include_endpoints_and_stay_in_rom() -> None:
    opened = np.array([0.0, -0.2])
    grasp = np.array([1.0, 0.8])
    targets = smooth_close_targets(
        opened,
        grasp,
        steps=5,
        joint_low=np.array([-0.5, -0.5]),
        joint_high=np.array([1.2, 1.0]),
    )
    np.testing.assert_allclose(targets[0], opened)
    np.testing.assert_allclose(targets[-1], grasp)
    assert np.all(np.diff(targets[:, 0]) >= 0.0)
    assert np.all(np.diff(targets[:, 1]) >= 0.0)


def _contact_summary() -> GraspContactSummary:
    return GraspContactSummary(
        palm_contact=True,
        contacted_fingers=frozenset({"index", "thumb"}),
        opposing_support=True,
        total_normal_force_n=1.2,
        peak_normal_force_n=0.7,
    )


def test_stability_gate_accepts_low_drift_opposing_multifinger_hold() -> None:
    samples = [
        GraspStabilitySample(
            time_s=index * 0.05,
            cube_position_in_palm_m=np.array([index * 0.0005, 0.0, 0.0]),
            relative_linear_speed_m_s=0.01,
            relative_angular_speed_rad_s=0.1,
            actuator_effort=0.4,
            contacts=_contact_summary(),
        )
        for index in range(6)
    ]
    result = evaluate_grasp_stability(
        samples,
        GraspStabilityThresholds(),
    )
    assert result.stable
    assert result.failures == ()
    assert result.max_drift_m == pytest.approx(0.0025)


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("relative_linear_speed_m_s", 0.06, "linear_speed"),
        ("relative_angular_speed_rad_s", 0.6, "angular_speed"),
        ("actuator_effort", 1.1, "actuator_effort"),
    ],
)
def test_stability_gate_reports_specific_failures(field, value, failure) -> None:
    sample = GraspStabilitySample(
        time_s=0.0,
        cube_position_in_palm_m=np.zeros(3),
        relative_linear_speed_m_s=0.0,
        relative_angular_speed_rad_s=0.0,
        actuator_effort=0.0,
        contacts=_contact_summary(),
    )
    changed = GraspStabilitySample(
        **{**sample.__dict__, field: value}
    )
    result = evaluate_grasp_stability(
        [sample, changed],
        GraspStabilityThresholds(max_actuator_effort=1.0),
    )
    assert not result.stable
    assert failure in result.failures
```

Add this MuJoCo-backed contact test using the real v2 reset:

```python
from orca_sim.grasping import GraspGeometryConfig, summarize_cube_contacts


def test_real_v2_contact_summary_classifies_initial_palm_contact() -> None:
    env = OrcaHandRightCubeOrientation(version="v2")
    try:
        env.reset(seed=1)
        summary = summarize_cube_contacts(
            env.model,
            env.data,
            cube_body_id=env._cube_body_id,
            geometry=GraspGeometryConfig(),
        )
        assert summary.palm_contact
        assert summary.total_normal_force_n >= 0.0
        assert summary.peak_normal_force_n >= 0.0
        assert np.isfinite(summary.total_normal_force_n)
        assert np.isfinite(summary.peak_normal_force_n)
    finally:
        env.close()
```

- [ ] **Step 2: Run the focused tests and verify new names fail**

Run:

```bash
cd /Users/jerry/Code/orca_sim
uv run pytest tests/test_grasping.py -v
```

Expected: import errors for the new stability interfaces.

- [ ] **Step 3: Implement the exact immutable data model and calculations**

Add these defaults and calculations to `grasping.py`:

```python
@dataclass(frozen=True)
class GraspGeometryConfig:
    palm_body_name: str = "right_R-Carpals_8d1f1041"
    finger_root_bodies: tuple[tuple[str, str], ...] = (
        ("pinky", "right_P-AP_f5e42b61"),
        ("ring", "right_M-AP_6ec59111"),
        ("middle", "right_M-AP_e04a96f2"),
        ("index", "right_I-AP-R_d95d02d1"),
        ("thumb", "right_T-TP-R_1c2b802d"),
    )
    opposing_normal_dot_max: float = -0.25


@dataclass(frozen=True)
class GraspStabilityThresholds:
    max_drift_m: float = 0.005
    max_linear_speed_m_s: float = 0.05
    max_angular_speed_rad_s: float = 0.5
    min_contacted_fingers: int = 2
    require_palm_contact: bool = True
    require_opposing_support: bool = True
    max_peak_normal_force_n: float = 8.0
    max_actuator_effort: float = 1.0


@dataclass(frozen=True)
class GraspContactSummary:
    palm_contact: bool
    contacted_fingers: frozenset[str]
    opposing_support: bool
    total_normal_force_n: float
    peak_normal_force_n: float


@dataclass(frozen=True)
class GraspStabilitySample:
    time_s: float
    cube_position_in_palm_m: np.ndarray
    relative_linear_speed_m_s: float
    relative_angular_speed_rad_s: float
    actuator_effort: float
    contacts: GraspContactSummary


@dataclass(frozen=True)
class GraspStabilityResult:
    stable: bool
    failures: tuple[str, ...]
    max_drift_m: float
    max_linear_speed_m_s: float
    max_angular_speed_rad_s: float
    contacted_fingers: tuple[str, ...]
    peak_normal_force_n: float
    peak_actuator_effort: float
```

Use `3u² - 2u³` for `smooth_close_targets`; require one-dimensional equal
shapes, `steps >= 2`, finite values, endpoints inside ROM, and return shape
`(steps, joints)`.

For contact classification, climb `model.body_parentid` from each non-cube
contact body to the configured palm and finger roots. Convert every contact
normal to a force direction on the cube: keep `contact.frame[:3]` when the cube
is `geom2`, negate it when the cube is `geom1`. `opposing_support` is true when
any normalized support-normal pair has dot product less than or equal to
`opposing_normal_dot_max`. Use `mujoco.mj_contactForce` and the normal-force
component at index zero.

For stability, reject non-finite samples first. Drift is the maximum Euclidean
distance from the first position in the supplied window. Report every failed
condition in deterministic order:
`non_finite`, `drift`, `linear_speed`, `angular_speed`, `palm_contact`,
`finger_contacts`, `opposing_support`, `normal_force`, `actuator_effort`.

- [ ] **Step 4: Run focused and regression tests**

```bash
cd /Users/jerry/Code/orca_sim
uv run pytest tests/test_grasping.py tests/test_cube_orientation_goals.py tests/test_drop_penalty.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
cd /Users/jerry/Code/orca_sim
git add src/orca_sim/grasping.py tests/test_grasping.py
git commit -m "Add grasp stability qualification"
```

---

### Task 3: Integrate bounded grasp preparation into cube-task reset

**Repository:** `orca_sim`

**Files:**
- Modify: `src/orca_sim/grasping.py`
- Modify: `src/orca_sim/task_envs.py:44-249,689-799`
- Create: `tests/test_grasp_reset.py`

**Interfaces:**
- Consumes: Task 1 artifact and placement, Task 2 trajectory/contact/stability interfaces.
- Produces: `GraspPreparationConfig`, `GraspPreparationResult`, `GraspPreparationError`, optional `grasp_artifact_path`, `grasp_reset_enabled`, `grasp_reset_attempts`, and `grasp_stability_thresholds` environment arguments, reset info keys prefixed `grasp_`, and this exact function:

```python
def execute_grasp_preparation(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_qpos_indices: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    cube_qpos_adr: int,
    cube_qvel_adr: int,
    cube_body_id: int,
    open_hand_qpos: np.ndarray,
    initial_cube_quat: np.ndarray,
    artifact: GraspPoseArtifact,
    config: GraspPreparationConfig,
    thresholds: GraspStabilityThresholds,
    geometry: GraspGeometryConfig,
    rng: np.random.Generator,
) -> GraspPreparationResult: ...
```

- [ ] **Step 1: Write failing reset integration tests**

Create `tests/test_grasp_reset.py` with a tiny valid artifact fixture and tests
that monkeypatch `execute_grasp_preparation` to isolate task-boundary behavior:

```python
from pathlib import Path

import numpy as np
import pytest

from orca_sim import OrcaHandRightCubeOrientation
from orca_sim.grasping import (
    GraspPoseArtifact,
    GraspPreparationError,
    GraspPreparationResult,
    GraspStabilityResult,
    write_grasp_artifact,
)


_FINGER_JOINTS = (
    "right_p-abd", "right_p-mcp", "right_p-pip",
    "right_r-abd", "right_r-mcp", "right_r-pip",
    "right_m-abd", "right_m-mcp", "right_m-pip",
    "right_i-abd", "right_i-mcp", "right_i-pip",
    "right_t-cmc", "right_t-abd", "right_t-mcp", "right_t-pip",
)


def _write_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "grasp_pose.json"
    write_grasp_artifact(
        path,
        GraspPoseArtifact(
            schema_version=1,
            joint_positions={name: 0.0 for name in _FINGER_JOINTS},
            close_duration_s=0.6,
            hold_duration_s=0.5,
            palm_site_name="right_palm_grasp_center",
            palm_offset_m=(0.0, 0.0, 0.0),
            validation_episodes=100,
            validation_stability_rate=0.95,
        ),
    )
    return path


def _stable_result() -> GraspPreparationResult:
    return GraspPreparationResult(
        attempts=1,
        measured_joint_positions=np.zeros(17),
        stability=GraspStabilityResult(
            stable=True,
            failures=(),
            max_drift_m=0.001,
            max_linear_speed_m_s=0.01,
            max_angular_speed_rad_s=0.1,
            contacted_fingers=("index", "thumb"),
            peak_normal_force_n=0.8,
            peak_actuator_effort=0.4,
        ),
    )


def test_grasp_preparation_finishes_before_task_clock_starts(
    tmp_path, monkeypatch
) -> None:
    artifact_path = _write_artifact(tmp_path)
    monkeypatch.setattr(
        "orca_sim.task_envs.execute_grasp_preparation",
        lambda **kwargs: _stable_result(),
    )
    env = OrcaHandRightCubeOrientation(
        version="v2",
        goal_mode="cube_orientation",
        grasp_reset_enabled=True,
        grasp_artifact_path=artifact_path,
    )
    try:
        _, info = env.reset(seed=7)
        assert info["elapsed_steps"] == 0
        assert info["task_elapsed_steps"] == 0
        assert info["grasp_stable"]
        assert info["grasp_attempts"] == 1
        assert info["grasp_contacted_fingers"] == ("index", "thumb")
    finally:
        env.close()


def test_failed_grasp_never_returns_policy_observation(tmp_path, monkeypatch) -> None:
    artifact_path = _write_artifact(tmp_path)
    monkeypatch.setattr(
        "orca_sim.task_envs.execute_grasp_preparation",
        lambda **kwargs: (_ for _ in ()).throw(
            GraspPreparationError(
                attempts=3,
                best_failures=("finger_contacts",),
                best_metrics={"max_drift_m": 0.002},
            )
        ),
    )
    env = OrcaHandRightCubeOrientation(
        version="v2",
        grasp_reset_enabled=True,
        grasp_artifact_path=artifact_path,
        grasp_reset_attempts=3,
    )
    try:
        with pytest.raises(GraspPreparationError, match="3 attempts"):
            env.reset(seed=7)
    finally:
        env.close()
```

- [ ] **Step 2: Run tests and verify constructor/interface failures**

```bash
cd /Users/jerry/Code/orca_sim
uv run pytest tests/test_grasp_reset.py -v
```

Expected: imports or unexpected constructor arguments fail.

- [ ] **Step 3: Implement preparation execution and structured failure**

Add these runtime contracts to `grasping.py`:

```python
@dataclass(frozen=True)
class GraspPreparationConfig:
    palm_site_name: str = "right_palm_grasp_center"
    palm_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    close_duration_s: float = 0.6
    hold_duration_s: float = 0.5
    stability_window_s: float = 0.25
    max_attempts: int = 3
    cube_position_jitter_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class GraspPreparationResult:
    attempts: int
    measured_joint_positions: np.ndarray
    stability: GraspStabilityResult


class GraspPreparationError(RuntimeError):
    def __init__(
        self,
        *,
        attempts: int,
        best_failures: tuple[str, ...],
        best_metrics: dict[str, float],
    ) -> None:
        self.attempts = attempts
        self.best_failures = best_failures
        self.best_metrics = dict(best_metrics)
        super().__init__(
            f"grasp preparation failed after {attempts} attempts: "
            f"{', '.join(best_failures)}"
        )
```

`execute_grasp_preparation(...)` uses the exact signature in the Interfaces
block. For each bounded attempt it must:

1. call `mujoco.mj_resetData`;
2. set the open hand qpos and zero qvel;
3. run `mj_forward`, place the cube at the named site plus sampled local jitter,
   set its quaternion, zero cube velocity, and run `mj_forward` again;
4. build the 17-element grasp command by retaining the fixed wrist and mapping
   every artifact finger name to its actuator;
5. validate every command against actuator ROM;
6. step every 2 ms physics interval through the smooth 0.6-second close;
7. hold for 0.5 seconds while collecting the final 0.25-second sample window;
8. return only a stable result; otherwise retain the best result by fewest
   failures, then lowest drift;
9. raise `GraspPreparationError` after `max_attempts`.

At every physics step set `data.ctrl` before `mujoco.mj_step`. Use measured
`data.qpos` for `measured_joint_positions`. Relative position and velocity are
computed in the palm site's current rotation frame. Treat non-finite simulator
state as a `non_finite` stability failure.

- [ ] **Step 4: Wire opt-in preparation into `OrcaHandRightCubeOrientation`**

Add constructor arguments with generic behavior preserved by default:

```python
grasp_reset_enabled: bool = False,
grasp_artifact_path: str | Path | None = None,
grasp_reset_attempts: int = 3,
grasp_stability_thresholds: GraspStabilityThresholds | None = None,
```

When enabled, require and load the artifact during construction, reject
artifact validation rate below `0.95`, and run preparation at the start of
`reset`. Only after preparation succeeds should task bags, targets, counters,
and progress baselines be initialized. Add these reset/step info keys:

```python
"grasp_stable": bool,
"grasp_attempts": int,
"grasp_hold_time_s": float,
"grasp_max_drift_m": float,
"grasp_max_linear_speed_m_s": float,
"grasp_max_angular_speed_rad_s": float,
"grasp_contacted_fingers": tuple[str, ...],
"grasp_peak_normal_force_n": float,
"grasp_peak_actuator_effort": float,
"grasp_artifact_hash": str | None,
```

Keep the existing reset path unchanged when disabled. Ensure the render call
still occurs only after the complete reset.

- [ ] **Step 5: Run integration and regression tests**

```bash
cd /Users/jerry/Code/orca_sim
uv run pytest tests/test_grasp_reset.py tests/test_cube_orientation_goals.py tests/test_registry.py -v
```

Expected: all pass, including legacy environments without an artifact.

- [ ] **Step 6: Commit Task 3**

```bash
cd /Users/jerry/Code/orca_sim
git add src/orca_sim/grasping.py src/orca_sim/task_envs.py tests/test_grasp_reset.py
git commit -m "Add stable grasp reset preparation"
```

---

### Task 4: Add curriculum-sized rotation targets and task diagnostics

**Repository:** `orca_sim`

**Files:**
- Modify: `src/orca_sim/task_envs.py:44-185,381-424,494-685`
- Modify: `tests/test_cube_orientation_goals.py`

**Interfaces:**
- Consumes: existing quaternion utilities and task goal logic.
- Produces: target policy `curriculum_fixed_turn`, constructor field `target_rotation_range_rad: tuple[float, float] | None`, reset sampling from the configured hand-X angle range, and info keys `orientation_error_delta_rad`, `cube_position_in_hand_m`, `drop_reason`, and current contact diagnostics.

- [ ] **Step 1: Write failing target and diagnostic tests**

Add:

```python
def test_curriculum_fixed_turn_samples_only_configured_hand_x_angles() -> None:
    env = OrcaHandRightCubeOrientation(
        version="v2",
        goal_mode="cube_orientation",
        target_policy="curriculum_fixed_turn",
        target_sequence_length=1,
        target_rotation_range_rad=(np.deg2rad(10.0), np.deg2rad(15.0)),
    )
    try:
        sampled = []
        for seed in range(20):
            _, info = env.reset(seed=seed)
            sampled.append(info["target_rotation_angle_rad"])
        assert min(sampled) >= np.deg2rad(10.0)
        assert max(sampled) <= np.deg2rad(15.0)
        assert len({round(value, 6) for value in sampled}) > 1
    finally:
        env.close()


def test_zero_angle_curriculum_target_supports_hold_stage() -> None:
    env = OrcaHandRightCubeOrientation(
        version="v2",
        goal_mode="cube_orientation",
        target_policy="curriculum_fixed_turn",
        target_sequence_length=1,
        target_rotation_range_rad=(0.0, 0.0),
    )
    try:
        _, info = env.reset(seed=1)
        assert info["target_rotation_angle_rad"] == pytest.approx(0.0)
        assert info["orientation_error_rad"] == pytest.approx(0.0)
        assert info["orientation_error_delta_rad"] == pytest.approx(0.0)
        assert info["cube_position_in_hand_m"].shape == (3,)
    finally:
        env.close()
```

Also extend the angular-progress test to assert that an error reduction from
`1.0` to `0.8` exposes `orientation_error_delta_rad == 0.2`.

- [ ] **Step 2: Run focused tests and verify policy validation fails**

```bash
cd /Users/jerry/Code/orca_sim
uv run pytest tests/test_cube_orientation_goals.py -v
```

Expected: `curriculum_fixed_turn` is rejected and diagnostic keys are absent.

- [ ] **Step 3: Implement target range validation and sampling**

Accept `curriculum_fixed_turn` only for `goal_mode="cube_orientation"` and
`target_sequence_length=1`. Normalize the range to two finite floats satisfying
`0 <= low <= high <= pi`. Require it for this target policy and reject it for
other policies to prevent ignored configuration.

In `_draw_next_orientation_target`, sample uniformly from `[low, high]`, rotate
the current cube quaternion about positive hand X, and report the sampled angle.
Use the current quaternion as source, including the zero-angle hold target.
Preserve all existing policies exactly.

Expose the orientation-error delta already maintained by `_update_task_status`.
Compute `cube_position_in_hand_m` as mount rotation transpose multiplied by
`cube_pos - mount_pos`. Reuse `summarize_cube_contacts` for current
`contacted_fingers`, `palm_contact`, and `opposing_support` info without changing
the observation vector. Clear `drop_reason` at reset; when the world-height
criterion triggers, set it to `"below_height"` before returning terminal info.

- [ ] **Step 4: Run all orientation and reward regressions**

```bash
cd /Users/jerry/Code/orca_sim
uv run pytest tests/test_cube_orientation_goals.py tests/test_drop_penalty.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit Task 4**

```bash
cd /Users/jerry/Code/orca_sim
git add src/orca_sim/task_envs.py tests/test_cube_orientation_goals.py
git commit -m "Add curriculum cube rotation targets"
```

---

### Task 5: Implement robust CEM grasp search and validation CLI

**Repository:** `orca_sim`

**Files:**
- Create: `src/orca_sim/grasp_search.py`
- Modify: `pyproject.toml:24-30`
- Create: `tests/test_grasp_search.py`

**Interfaces:**
- Consumes: `execute_grasp_preparation`, v2 task environment, 16 finger ROM bounds, and artifact I/O.
- Produces: `CEMSearchConfig`, `GraspPerturbation`, `CandidateEvaluation`, `CEMSearchResult`, CLI `orca-grasp {search,validate}`, and these exact functions:

```python
def robust_candidate_key(
    evaluation: CandidateEvaluation,
) -> tuple[float, float, float]: ...

def search_cem(
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    evaluator: Callable[[np.ndarray], CandidateEvaluation],
    config: CEMSearchConfig,
) -> CEMSearchResult: ...

def evaluate_candidate(
    candidate: np.ndarray,
    *,
    joint_names: tuple[str, ...],
    scenarios: Sequence[GraspPerturbation],
    env_factory: Callable[[], OrcaHandRightCubeOrientation],
    artifact_template: GraspPoseArtifact,
    preparation_config: GraspPreparationConfig,
    thresholds: GraspStabilityThresholds,
) -> CandidateEvaluation: ...
```

- [ ] **Step 1: Write failing deterministic CEM and robust-ranking tests**

Create tests with a cheap mathematical evaluator so the optimizer test does not
run MuJoCo populations:

```python
import json

import numpy as np
import pytest

from orca_sim.grasp_search import (
    CEMSearchConfig,
    CandidateEvaluation,
    robust_candidate_key,
    search_cem,
)


def test_cem_is_deterministic_and_converges_inside_bounds() -> None:
    target = np.array([0.25, -0.4])

    def evaluator(candidate: np.ndarray) -> CandidateEvaluation:
        score = -float(np.sum((candidate - target) ** 2))
        return CandidateEvaluation(
            joint_positions=candidate.copy(),
            stability_rate=1.0,
            robust_score=score,
            median_drift_m=0.0,
            failures={},
        )

    config = CEMSearchConfig(
        population=64,
        elite_fraction=0.125,
        iterations=8,
        initial_std_fraction=0.4,
        min_std_rad=0.005,
        seed=7,
    )
    first = search_cem(
        lower=np.array([-1.0, -1.0]),
        upper=np.array([1.0, 1.0]),
        evaluator=evaluator,
        config=config,
    )
    second = search_cem(
        lower=np.array([-1.0, -1.0]),
        upper=np.array([1.0, 1.0]),
        evaluator=evaluator,
        config=config,
    )
    np.testing.assert_allclose(first.best.joint_positions, second.best.joint_positions)
    np.testing.assert_allclose(first.best.joint_positions, target, atol=0.04)


def test_robust_ranking_prefers_validation_rate_before_mean_score() -> None:
    stable = CandidateEvaluation(
        joint_positions=np.zeros(2),
        stability_rate=0.96,
        robust_score=1.0,
        median_drift_m=0.003,
        failures={"drop": 4},
    )
    brittle = CandidateEvaluation(
        joint_positions=np.ones(2),
        stability_rate=0.80,
        robust_score=100.0,
        median_drift_m=0.001,
        failures={"drop": 20},
    )
    assert max((stable, brittle), key=robust_candidate_key) is stable
```

Add CLI parser tests asserting exact defaults: population `128`, elite fraction
`0.1`, iterations `8`, validation episodes `100`, required stability `0.95`,
scenario mode `perturbed`, and separate search/validation seed ranges. The
`validate` parser accepts `--scenario-mode {nominal,perturbed}`.

- [ ] **Step 2: Run tests and verify module import failure**

```bash
cd /Users/jerry/Code/orca_sim
uv run pytest tests/test_grasp_search.py -v
```

Expected: collection fails because `grasp_search` does not exist.

- [ ] **Step 3: Implement dependency-free CEM and robust score ordering**

Use these immutable types:

```python
@dataclass(frozen=True)
class CEMSearchConfig:
    population: int = 128
    elite_fraction: float = 0.1
    iterations: int = 8
    initial_std_fraction: float = 0.3
    min_std_rad: float = 0.02
    seed: int = 1


@dataclass(frozen=True)
class GraspPerturbation:
    palm_offset_m: tuple[float, float, float]
    cube_rotation_rad: tuple[float, float, float]
    mass_scale: float
    friction_scale: float
    actuator_gain_scale: float


@dataclass(frozen=True)
class CandidateEvaluation:
    joint_positions: np.ndarray
    stability_rate: float
    robust_score: float
    median_drift_m: float
    failures: dict[str, int]


@dataclass(frozen=True)
class CEMSearchResult:
    best: CandidateEvaluation
    iteration_best_scores: tuple[float, ...]
    evaluated_candidates: int
```

Validate all configuration values. Initialize mean at the midpoint of bounded
finger ROM and standard deviation as `initial_std_fraction * (upper-lower)`.
Sample with `np.random.default_rng(seed)`, clip every candidate to bounds,
evaluate sequentially in the pure function, sort by
`(stability_rate, robust_score, -median_drift_m)`, and update mean/std from the
top `ceil(population * elite_fraction)` candidates. Clamp std to
`min_std_rad`. Return the best candidate seen across all iterations.

- [ ] **Step 4: Implement simulator candidate evaluation and CLI**

Use deterministic scenario generation with disjoint defaults:

- search seeds `1-32` per candidate;
- held-out validation seeds `10001-10100`;
- position jitter independently uniform within ±3 mm in palm X/Y and ±2 mm in
  palm Z;
- cube rotation jitter independently uniform within ±5 degrees;
- mass scale uniform `[0.8, 1.2]`;
- friction scale uniform `[0.7, 1.3]`;
- actuator gain scale uniform `[0.9, 1.1]`.

For each scenario call the real preparation function. Candidate
`stability_rate` is stable attempts divided by scenarios. Define per-scenario
score as:

```text
1.0
- 40 * drift_m
- 2 * max_linear_speed_m_s
- 0.2 * max_angular_speed_rad_s
- 0.02 * peak_normal_force_n
- 0.05 * peak_actuator_effort
```

Failed scenarios score `-1.0`. `robust_score` is the 10th percentile of
scenario scores. The report must include search config, perturbation ranges,
iteration history, best candidate, held-out metrics, failure histogram, and
artifact hash.

Candidate evaluation must snapshot cube body mass, cube geom friction,
`model.actuator_gainprm`, and `model.actuator_biasprm` before perturbation.
Apply mass/friction scales and scale position-actuator gain/bias consistently,
then restore every array in a `finally` block before the next scenario. This
prevents perturbations from accumulating across candidates.

Register:

```toml
[project.scripts]
orca-grasp = "orca_sim.grasp_search:main"
```

The `search` subcommand writes an artifact only when held-out stability is at
least `--required-stability` (default `0.95`); otherwise it writes the report,
prints the best rate, and exits non-zero. The `validate` subcommand verifies the
artifact hash, runs exactly the requested held-out episodes, writes a report,
and exits non-zero below the threshold. `--scenario-mode nominal` disables all
perturbations and `--hold-duration-s` overrides artifact hold time for the
100/100 two-second passive-hold acceptance; `perturbed` uses the held-out
randomization ranges above.

- [ ] **Step 5: Run search unit tests and a two-candidate MuJoCo smoke**

```bash
cd /Users/jerry/Code/orca_sim
uv run pytest tests/test_grasp_search.py -v
uv run orca-grasp search --population 2 --iterations 1 --search-scenarios 1 --validation-episodes 1 --output /private/tmp/orca-grasp-smoke.json --report /private/tmp/orca-grasp-smoke-report.json --allow-below-threshold
```

Expected: tests pass; smoke exits zero only because
`--allow-below-threshold` suppresses the acceptance exit for local mechanics,
and both JSON files parse successfully.

- [ ] **Step 6: Commit Task 5**

```bash
cd /Users/jerry/Code/orca_sim
git add src/orca_sim/grasp_search.py pyproject.toml tests/test_grasp_search.py
git commit -m "Add robust grasp pose search"
```

---

### Task 6: Add grasp-aware reward and a safe residual-action envelope

**Repository:** `orca_train`

**Files:**
- Create: `src/orca_train/grasp_reward.py`
- Modify: `src/orca_train/state_env.py:12-194`
- Create: `tests/test_grasp_reward.py`
- Modify: `tests/test_state_env.py`

**Interfaces:**
- Consumes: sim info fields from Tasks 3-4 and verified `GraspPoseArtifact`.
- Produces: `GraspRewardConfig`, `GraspRewardInputs`, `GraspRewardBreakdown`, state-env args `grasp_joint_positions`, `safety_envelope_degrees`, and `grasp_reward_config`, and these exact functions:

```python
def minimum_safe_drop_penalty(config: GraspRewardConfig) -> float: ...

def compute_grasp_reward(
    inputs: GraspRewardInputs,
    config: GraspRewardConfig,
) -> GraspRewardBreakdown: ...
```

- [ ] **Step 1: Write failing reward exploit and dominance tests**

Create `tests/test_grasp_reward.py`:

```python
import numpy as np
import pytest

from orca_train.grasp_reward import (
    GraspRewardConfig,
    GraspRewardInputs,
    compute_grasp_reward,
    minimum_safe_drop_penalty,
)


def _inputs(**changes) -> GraspRewardInputs:
    values = {
        "orientation_error_delta_rad": 0.2,
        "cube_drift_m": 0.001,
        "linear_speed_m_s": 0.01,
        "angular_speed_rad_s": 0.1,
        "action": np.zeros(16),
        "previous_action": np.zeros(16),
        "target_completed": False,
        "dropped": False,
    }
    values.update(changes)
    return GraspRewardInputs(**values)


def test_falling_orientation_improvement_cannot_earn_positive_reward() -> None:
    reward = compute_grasp_reward(
        _inputs(
            cube_drift_m=0.03,
            linear_speed_m_s=0.8,
            angular_speed_rad_s=12.0,
        ),
        GraspRewardConfig(drop_penalty=30.0),
    )
    assert reward.grasp_quality == 0.0
    assert reward.orientation_progress_reward == 0.0
    assert reward.total < 0.0


def test_negative_orientation_progress_is_not_hidden_by_quality_gate() -> None:
    reward = compute_grasp_reward(
        _inputs(
            orientation_error_delta_rad=-0.2,
            cube_drift_m=0.03,
            linear_speed_m_s=0.8,
        ),
        GraspRewardConfig(drop_penalty=30.0),
    )
    assert reward.orientation_progress_reward == pytest.approx(-1.0)


def test_drop_penalty_exceeds_all_possible_positive_episode_reward() -> None:
    config = GraspRewardConfig(
        progress_reward_scale=5.0,
        success_bonus=10.0,
        drop_penalty=30.0,
    )
    assert config.drop_penalty >= minimum_safe_drop_penalty(config)
    with pytest.raises(ValueError, match="drop_penalty"):
        GraspRewardConfig(
            progress_reward_scale=5.0,
            success_bonus=10.0,
            drop_penalty=20.0,
        )
```

- [ ] **Step 2: Run reward tests and verify module import failure**

```bash
cd /Users/jerry/Code/orca_train
uv run pytest tests/test_grasp_reward.py -v
```

Expected: collection fails because `grasp_reward` does not exist.

- [ ] **Step 3: Implement the exact reward formula**

Use these defaults:

```python
@dataclass(frozen=True)
class GraspRewardConfig:
    progress_reward_scale: float = 5.0
    success_bonus: float = 10.0
    drop_penalty: float = 30.0
    step_penalty: float = 0.01
    max_reward_drift_m: float = 0.01
    max_reward_linear_speed_m_s: float = 0.10
    max_reward_angular_speed_rad_s: float = 1.0
    drift_penalty_scale: float = 20.0
    linear_speed_penalty_scale: float = 0.5
    angular_speed_penalty_scale: float = 0.05
    action_rate_penalty_scale: float = 0.02
    saturation_penalty_scale: float = 0.05
    saturation_threshold: float = 0.95


@dataclass(frozen=True)
class GraspRewardInputs:
    orientation_error_delta_rad: float
    cube_drift_m: float
    linear_speed_m_s: float
    angular_speed_rad_s: float
    action: np.ndarray
    previous_action: np.ndarray
    target_completed: bool
    dropped: bool


@dataclass(frozen=True)
class GraspRewardBreakdown:
    total: float
    grasp_quality: float
    orientation_progress_reward: float
    drift_penalty: float
    motion_penalty: float
    action_rate_penalty: float
    saturation_penalty: float
    success_reward: float
    drop_reward: float
```

Compute each quality factor as `clip(1 - value/limit, 0, 1)` and multiply the
drift, linear-speed, and angular-speed factors. Multiply positive orientation
progress by quality; retain negative orientation progress at full scale.
Penalties are non-negative magnitudes subtracted from total. Saturation is the
fraction of action elements whose absolute value is at least the threshold.
`minimum_safe_drop_penalty` returns
`progress_reward_scale * pi + success_bonus + 1.0`; validate the configured
drop penalty against it in `__post_init__`.

- [ ] **Step 4: Write failing state-env safety-envelope and takeover tests**

Extend the fake environment info to include the Task 4 fields, then add:

```python
def test_state_action_is_clamped_to_grasp_safety_envelope() -> None:
    base = _FakeStateCubeEnv()
    grasp = base.data.qpos.astype(np.float32).copy()
    env = OrcaStateCubeEnv(
        env=base,
        max_delta_degrees=3.0,
        grasp_joint_positions=grasp,
        safety_envelope_degrees=5.0,
    )
    env.reset()
    for _ in range(10):
        env.step(np.ones(17, dtype=np.float32))
    assert np.all(base.last_action <= grasp + np.deg2rad(5.0) + 1e-7)


def test_policy_takeover_target_starts_at_measured_grasp() -> None:
    base = _FakeStateCubeEnv()
    grasp = base.data.qpos.astype(np.float32).copy()
    env = OrcaStateCubeEnv(
        env=base,
        grasp_joint_positions=grasp,
        safety_envelope_degrees=5.0,
    )
    env.reset()
    measured = base.data.qpos.copy()
    env.step(np.zeros(17, dtype=np.float32))
    np.testing.assert_allclose(base.last_action, measured)
```

- [ ] **Step 5: Integrate reward and safety metrics into `OrcaStateCubeEnv`**

Accept `grasp_joint_positions: np.ndarray | None`,
`safety_envelope_degrees: float | None`, and
`grasp_reward_config: GraspRewardConfig | None`. Validate the grasp shape is
`(17,)`, the envelope is positive, and grasp positions are inside ROM. Clamp
the measured-delta target first to actuator ROM and then to
`measured_grasp_at_reset ± envelope`. Use the artifact pose to validate that
the measured grasp is within the same safety envelope, but use the
measured center for control so a zero normalized action cannot jump. Fixed
joints remain unchanged.

At reset store the measured target, initial `cube_position_in_hand_m`, and a
zero previous normalized action. At step compute drift from that anchor,
construct `GraspRewardInputs`, replace the base reward only when reward config
is present, and add all `GraspRewardBreakdown` fields plus
`action_saturation_rate` and `action_rate_l2` to info. Update previous action
after calculating reward. Do not change the 47-element observation.

- [ ] **Step 6: Run focused tests and state regressions**

```bash
cd /Users/jerry/Code/orca_train
uv run pytest tests/test_grasp_reward.py tests/test_state_env.py -v
```

Expected: all pass and state shape remains `(47,)`.

- [ ] **Step 7: Commit Task 6**

```bash
cd /Users/jerry/Code/orca_train
git add src/orca_train/grasp_reward.py src/orca_train/state_env.py tests/test_grasp_reward.py tests/test_state_env.py
git commit -m "Add grasp-aware state control rewards"
```

---

### Task 7: Implement the five-stage curriculum state machine

**Repository:** `orca_train`

**Files:**
- Create: `src/orca_train/curriculum.py`
- Create: `tests/test_curriculum.py`

**Interfaces:**
- Consumes: evaluation mappings containing `success_rate` and `drop_rate`.
- Produces: `CurriculumStage`, `CurriculumState`, `GRASP_CURRICULUM`, and these exact functions:

```python
def stage_task_kwargs(stage: CurriculumStage) -> dict[str, object]: ...

def update_curriculum(
    state: CurriculumState,
    evaluation: Mapping[str, float],
    *,
    required_streak: int = 2,
) -> CurriculumState: ...
```

- [ ] **Step 1: Write failing stage-definition and promotion tests**

Create:

```python
import numpy as np

from orca_train.curriculum import (
    GRASP_CURRICULUM,
    CurriculumState,
    stage_task_kwargs,
    update_curriculum,
)


def test_grasp_curriculum_has_approved_stage_angles_and_thresholds() -> None:
    assert [stage.name for stage in GRASP_CURRICULUM] == [
        "hold", "turn_10_15", "turn_30", "turn_60", "turn_90"
    ]
    assert GRASP_CURRICULUM[0].target_rotation_range_rad == (0.0, 0.0)
    assert GRASP_CURRICULUM[1].target_rotation_range_rad == (
        np.deg2rad(10.0), np.deg2rad(15.0)
    )
    assert GRASP_CURRICULUM[-1].target_rotation_range_rad == (
        np.pi / 2.0, np.pi / 2.0
    )
    assert GRASP_CURRICULUM[0].min_success_rate == 0.99
    assert GRASP_CURRICULUM[0].max_drop_rate == 0.01
    assert all(stage.min_success_rate == 0.8 for stage in GRASP_CURRICULUM[1:])
    assert all(stage.max_drop_rate == 0.05 for stage in GRASP_CURRICULUM[1:])


def test_promotion_requires_two_consecutive_passing_evaluations() -> None:
    state = CurriculumState()
    first = update_curriculum(state, {"success_rate": 1.0, "drop_rate": 0.0})
    assert first.stage_index == 0
    assert first.promotion_streak == 1
    second = update_curriculum(first, {"success_rate": 0.99, "drop_rate": 0.01})
    assert second.stage_index == 1
    assert second.promotion_streak == 0


def test_failed_evaluation_resets_streak_and_final_stage_does_not_advance() -> None:
    state = CurriculumState(stage_index=0, promotion_streak=1)
    failed = update_curriculum(state, {"success_rate": 0.98, "drop_rate": 0.0})
    assert failed == CurriculumState(stage_index=0, promotion_streak=0)
    terminal = CurriculumState(stage_index=4, promotion_streak=7)
    assert update_curriculum(
        terminal, {"success_rate": 1.0, "drop_rate": 0.0}
    ) == terminal
```

- [ ] **Step 2: Run the new tests and verify module import failure**

```bash
cd /Users/jerry/Code/orca_train
uv run pytest tests/test_curriculum.py -v
```

Expected: collection fails because `curriculum` does not exist.

- [ ] **Step 3: Implement immutable stages and deterministic promotion**

Define:

```python
@dataclass(frozen=True)
class CurriculumStage:
    name: str
    target_rotation_range_rad: tuple[float, float]
    max_task_duration_s: float
    success_hold_duration_s: float
    safety_envelope_degrees: float
    min_success_rate: float
    max_drop_rate: float


@dataclass(frozen=True)
class CurriculumState:
    stage_index: int = 0
    promotion_streak: int = 0
```

Use these exact stage values:

```python
GRASP_CURRICULUM = (
    CurriculumStage("hold", (0.0, 0.0), 2.0, 2.0, 5.0, 0.99, 0.01),
    CurriculumStage("turn_10_15", (np.deg2rad(10.0), np.deg2rad(15.0)), 4.0, 0.16, 7.5, 0.8, 0.05),
    CurriculumStage("turn_30", (np.deg2rad(30.0), np.deg2rad(30.0)), 6.0, 0.16, 10.0, 0.8, 0.05),
    CurriculumStage("turn_60", (np.deg2rad(60.0), np.deg2rad(60.0)), 8.0, 0.16, 15.0, 0.8, 0.05),
    CurriculumStage("turn_90", (np.pi / 2.0, np.pi / 2.0), 8.0, 0.16, 25.0, 0.8, 0.05),
)
```

`stage_task_kwargs` returns explicit kwargs for
`target_policy="curriculum_fixed_turn"`, `target_sequence_length=1`, the
rotation range, duration, hold duration, and safety envelope. Validate state
indices. `update_curriculum` requires finite rates in `[0,1]`, increments the
streak only when both thresholds pass, promotes at streak two, resets the
streak on failure, and never advances beyond the final stage.

- [ ] **Step 4: Run tests**

```bash
cd /Users/jerry/Code/orca_train
uv run pytest tests/test_curriculum.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit Task 7**

```bash
cd /Users/jerry/Code/orca_train
git add src/orca_train/curriculum.py tests/test_curriculum.py
git commit -m "Add grasp rotation curriculum"
```

---

### Task 8: Persist grasp and curriculum compatibility in checkpoints

**Repository:** `orca_train`

**Files:**
- Modify: `src/orca_train/state_agent.py:1-169`
- Modify: `tests/test_state_agent.py`

**Interfaces:**
- Consumes: mapping metadata supplied by the training loop.
- Produces: `StateCheckpointMetadata` and these exact extensions while preserving integer step return:

```python
def read_state_checkpoint_metadata(
    path: str | Path,
) -> StateCheckpointMetadata: ...

def StateAgent.save(
    self,
    path: str | Path,
    step: int,
    *,
    run_metadata: StateCheckpointMetadata | None = None,
) -> None: ...

def StateAgent.load(
    self,
    path: str | Path,
    *,
    expected_run_metadata: StateCheckpointMetadata | None = None,
) -> int: ...
```

- [ ] **Step 1: Write failing checkpoint metadata tests**

Add:

```python
from orca_train.state_agent import (
    StateCheckpointMetadata,
    read_state_checkpoint_metadata,
)


def test_state_checkpoint_round_trips_grasp_and_curriculum_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "metadata.pt"
    metadata = StateCheckpointMetadata(
        grasp_artifact_hash="abc123",
        curriculum_version=1,
        curriculum_stage=2,
        promotion_streak=1,
    )
    agent = StateAgent(47, 16, "cpu", StateAgentConfig(hidden_dim=32))
    agent.save(checkpoint, step=7, run_metadata=metadata)
    assert read_state_checkpoint_metadata(checkpoint) == metadata
    restored = StateAgent(47, 16, "cpu", StateAgentConfig(hidden_dim=32))
    assert restored.load(checkpoint, expected_run_metadata=metadata) == 7


def test_state_checkpoint_rejects_different_grasp_hash(tmp_path) -> None:
    checkpoint = tmp_path / "metadata.pt"
    agent = StateAgent(47, 16, "cpu", StateAgentConfig(hidden_dim=32))
    agent.save(
        checkpoint,
        step=7,
        run_metadata=StateCheckpointMetadata("hash-a", 1, 0, 0),
    )
    with pytest.raises(ValueError, match="grasp_artifact_hash"):
        StateAgent(47, 16, "cpu", StateAgentConfig(hidden_dim=32)).load(
            checkpoint,
            expected_run_metadata=StateCheckpointMetadata("hash-b", 1, 0, 0),
        )
```

- [ ] **Step 2: Run focused tests and verify interface failures**

```bash
cd /Users/jerry/Code/orca_train
uv run pytest tests/test_state_agent.py -v
```

Expected: imports and extended save/load keyword arguments fail.

- [ ] **Step 3: Implement metadata serialization and compatibility rules**

Define:

```python
@dataclass(frozen=True)
class StateCheckpointMetadata:
    grasp_artifact_hash: str
    curriculum_version: int
    curriculum_stage: int
    promotion_streak: int
```

Validate non-empty hash, version `1`, stage in `0..4`, and non-negative streak.
Save `asdict(run_metadata)` under `run_metadata`. The read helper uses
`torch.load(..., map_location="cpu", weights_only=True)`, requires the mapping,
and constructs the dataclass. When expected metadata is supplied, require exact
hash and curriculum version; use the checkpoint's stage/streak as resume state
rather than requiring them to match a newly initialized state. Retain all
existing dimension and hidden-size validation and keep `load` returning the
integer step.

- [ ] **Step 4: Run all agent tests**

```bash
cd /Users/jerry/Code/orca_train
uv run pytest tests/test_state_agent.py -v
```

Expected: all pass, including checkpoints without run metadata when no expected
metadata is supplied.

- [ ] **Step 5: Commit Task 8**

```bash
cd /Users/jerry/Code/orca_train
git add src/orca_train/state_agent.py tests/test_state_agent.py
git commit -m "Validate grasp curriculum checkpoints"
```

---

### Task 9: Integrate safe seeding, promotion, metrics, and resume into state training

**Repository:** `orca_train`

**Files:**
- Modify: `src/orca_train/train_state.py:19-454`
- Modify: `tests/test_train_state.py`

**Interfaces:**
- Consumes: Tasks 3-8, a validated grasp artifact path, and the existing replay/agent loop.
- Produces: preset `grasp_curriculum`, episode-boundary promotion/rebuild, rich evaluation metrics, exact CLI options, compatible resume, and these exact helpers:

```python
def sample_safe_seed_action(
    rng: np.random.Generator,
    action_dim: int,
    *,
    std: float,
    clip: float,
) -> np.ndarray: ...

def build_state_env(
    config: StateTrainConfig,
    stage: CurriculumStage,
) -> OrcaStateCubeEnv: ...
```

- [ ] **Step 1: Write failing config, CLI, and safe-seeding tests**

Add tests asserting:

```python
from orca_train.train_state import build_state_env, sample_safe_seed_action


def test_grasp_curriculum_requires_validated_artifact(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="grasp artifact"):
        resolve_preset(
            StateTrainConfig(preset="grasp_curriculum", grasp_artifact=missing)
        )


def test_safe_seed_actions_are_small_zero_centered_residuals() -> None:
    rng = np.random.default_rng(7)
    actions = np.stack(
        [sample_safe_seed_action(rng, 16, std=0.10, clip=0.25) for _ in range(500)]
    )
    assert np.max(np.abs(actions)) <= 0.25
    assert np.abs(actions.mean()) < 0.02
    assert 0.07 < actions.std() < 0.11


def test_grasp_curriculum_cli_parses_artifact_and_safe_seed_options(tmp_path) -> None:
    artifact = tmp_path / "grasp.json"
    config = parse_args(
        [
            "--preset", "grasp_curriculum",
            "--grasp-artifact", str(artifact),
            "--safe-seed-std", "0.08",
            "--safe-seed-clip", "0.2",
            "--eval-episodes", "100",
        ]
    )
    assert config.grasp_artifact == artifact
    assert config.safe_seed_std == pytest.approx(0.08)
    assert config.safe_seed_clip == pytest.approx(0.2)
    assert config.eval_episodes == 100
```

Use a real temporary artifact fixture for `resolve_preset` success tests and
assert the resolved drop penalty is `30.0`, target policy is
`curriculum_fixed_turn`, and stage index is zero.

- [ ] **Step 2: Run focused tests and verify missing fields/functions fail**

```bash
cd /Users/jerry/Code/orca_train
uv run pytest tests/test_train_state.py -v
```

Expected: unknown preset or missing dataclass fields/functions fail.

- [ ] **Step 3: Extend configuration and environment construction**

Add to `StateTrainConfig`:

```python
grasp_artifact: Path | None = None
grasp_reset_attempts: int = 3
safe_seed_std: float = 0.10
safe_seed_clip: float = 0.25
curriculum_version: int = 1
curriculum_stage: int = 0
promotion_streak: int = 0
```

Add `grasp_curriculum` to CLI/preset choices. In `resolve_preset`, load and hash
validate the artifact, require held-out stability at least `0.95`, set stage
zero values from `GRASP_CURRICULUM`, use `fix_wrist=True`, drop penalty `30.0`,
reward mode `angular_progress`, and a distinct default output directory
`runs/state_cube_grasp_curriculum_seed<seed>`.

Implement:

```python
def sample_safe_seed_action(
    rng: np.random.Generator,
    action_dim: int,
    *,
    std: float,
    clip: float,
) -> np.ndarray:
    return np.clip(
        rng.normal(0.0, std, size=action_dim), -clip, clip
    ).astype(np.float32)
```

`build_state_env(config, stage)` must load the artifact once, map its 16 finger
joint positions into the model's 17-actuator order while retaining measured
wrist position, pass grasp reset settings to `OrcaStateCubeEnv`, pass stage
task kwargs to the base task, create `GraspRewardConfig` from resolved reward
values, and use the stage safety envelope.

- [ ] **Step 4: Extend evaluation with approved grasp metrics**

Accumulate and return:

```python
"mean_grasp_max_drift_m"
"mean_grasp_hold_time_s"
"mean_contacted_finger_count"
"mean_action_saturation_rate"
"mean_action_rate_l2"
"drop_reasons"
"curriculum_stage"
"curriculum_stage_name"
```

Use reset info for preparation drift/contact metrics and step info for action
metrics. `drop_reasons` counts `info["drop_reason"]` values, with `"unknown"`
used only when a legacy fake environment reports a drop without the new key.
Preserve all existing evaluation keys.

- [ ] **Step 5: Write failing episode-boundary promotion and resume tests**

Extend `_TinyStateEnv` or add a curriculum fake whose constructor records the
stage. Configure two consecutive passing evaluations. Assert:

- stage zero is used until an episode boundary;
- two passing evaluations advance to stage one;
- both train and eval environments close and are rebuilt with stage one;
- the emitted `curriculum_transition` event contains `from_stage=0` and
  `to_stage=1`;
- the checkpoint contains the artifact hash, stage one, and streak zero;
- resume reconstructs stage one before the first reset;
- a different artifact hash raises before any environment is created.

- [ ] **Step 6: Implement boundary-safe evaluation and promotion**

Change evaluation scheduling so a due evaluation runs immediately after a
training episode ends. Maintain `next_eval_step`; if several intervals pass in
one episode, run one evaluation and advance `next_eval_step` until it exceeds
the current step. This guarantees an environment is never replaced mid-episode.

After evaluation:

1. update `CurriculumState`;
2. write the evaluation event with current stage;
3. when stage changes, write `curriculum_transition`, close both environments,
   build both with the new stage, reset the n-step accumulator, and reset the
   new training environment;
4. save checkpoint metadata with the artifact hash and new state.

On resume, read checkpoint metadata before constructing environments. Require
hash/version compatibility, initialize curriculum from saved stage/streak,
then load agent weights and optimizers. During seed steps use
`sample_safe_seed_action` for `grasp_curriculum`; retain legacy uniform seeding
for legacy presets.

Add per-episode metrics for grasp drift, contacted-finger count, action
saturation/rate, stage, and termination reason.

- [ ] **Step 7: Add exact CLI arguments**

Add:

```text
--grasp-artifact PATH
--grasp-reset-attempts INT          default 3
--safe-seed-std FLOAT               default 0.10
--safe-seed-clip FLOAT              default 0.25
--curriculum-stage INT              default 0
--promotion-streak INT              default 0
```

Reject stage/streak overrides when `--resume` is used; checkpoint metadata is
authoritative. Validate positive attempts, `0 < safe_seed_std <= safe_seed_clip
<= 1`, stage `0..4`, and non-negative streak.

- [ ] **Step 8: Run all state-training tests**

```bash
cd /Users/jerry/Code/orca_train
uv run pytest tests/test_train_state.py tests/test_state_agent.py tests/test_curriculum.py tests/test_state_env.py tests/test_grasp_reward.py -v
```

Expected: all pass.

- [ ] **Step 9: Commit Task 9**

```bash
cd /Users/jerry/Code/orca_train
git add src/orca_train/train_state.py tests/test_train_state.py
git commit -m "Train with a stable grasp curriculum"
```

---

### Task 10: Add acceptance commands and operator documentation

**Repositories:** `orca_sim`, `orca_train`

**Files:**
- Modify: `orca_sim/src/orca_sim/grasp_search.py`
- Modify: `orca_sim/tests/test_grasp_search.py`
- Modify: `orca_train/README.md`

**Interfaces:**
- Consumes: complete search, validation, and training CLIs.
- Produces: machine-readable acceptance summaries and exact Spark-ready operator commands.

- [ ] **Step 1: Write a failing acceptance-summary test**

Add a CLI test that invokes `validate` with a fake evaluator and asserts the
report contains:

```python
assert report["acceptance"] == {
    "episodes": 100,
    "stable_episodes": 95,
    "stability_rate": 0.95,
    "required_stability_rate": 0.95,
    "passed": True,
}
assert set(report["metrics"]) >= {
    "median_drift_m",
    "max_drift_m",
    "mean_linear_speed_m_s",
    "mean_angular_speed_rad_s",
    "contacted_finger_histogram",
    "failure_histogram",
}
```

- [ ] **Step 2: Run the test and verify the report schema fails**

```bash
cd /Users/jerry/Code/orca_sim
uv run pytest tests/test_grasp_search.py -v
```

Expected: assertion failure for missing acceptance/metrics keys.

- [ ] **Step 3: Implement the stable report schema and exit behavior**

Sort all histogram keys, emit only finite JSON numbers, include artifact hash
and exact seed range, and return exit code zero only when `passed` is true.
`--allow-below-threshold` remains restricted to the `search` smoke path and is
not accepted by `validate`.

- [ ] **Step 4: Document exact local and Spark workflow**

Update `orca_train/README.md` with these commands and explain every acceptance
gate:

```bash
# Local fast tests only
cd /Users/jerry/Code/orca_sim
uv run pytest tests/
cd /Users/jerry/Code/orca_train
uv run pytest tests/

# Spark grasp search
uv run orca-grasp search \
  --population 128 \
  --iterations 8 \
  --search-scenarios 32 \
  --validation-episodes 100 \
  --required-stability 0.95 \
  --seed 1 \
  --output runs/grasp_search_seed1/grasp_pose.json \
  --report runs/grasp_search_seed1/search_report.json

# Independent held-out validation
uv run orca-grasp validate \
  --artifact runs/grasp_search_seed1/grasp_pose.json \
  --episodes 100 \
  --seed 20001 \
  --scenario-mode perturbed \
  --required-stability 0.95 \
  --report runs/grasp_search_seed1/validation_report.json

# Nominal 100/100 two-second zero-action hold acceptance
uv run orca-grasp validate \
  --artifact runs/grasp_search_seed1/grasp_pose.json \
  --episodes 100 \
  --seed 30001 \
  --scenario-mode nominal \
  --required-stability 1.0 \
  --hold-duration-s 2.0 \
  --report runs/grasp_search_seed1/nominal_validation_report.json

# Stage-0 curriculum training, which promotes automatically
uv run orca-train-state \
  --preset grasp_curriculum \
  --grasp-artifact ../orca_sim/runs/grasp_search_seed1/grasp_pose.json \
  --total-steps 1000000 \
  --eval-episodes 100 \
  --device cuda \
  --seed 1 \
  --output-dir runs/state_cube_grasp_curriculum_seed1
```

Document the physical equivalent: place the cube at the marked palm center,
execute the same 0.6-second position trajectory, hold, and qualify with tracked
cube motion plus motor current. Training must not start if search, perturbed
validation, or nominal 100/100 validation exits non-zero.

- [ ] **Step 5: Run README command parser smoke and tests**

```bash
cd /Users/jerry/Code/orca_sim
uv run orca-grasp --help
uv run orca-grasp search --help
uv run orca-grasp validate --help
uv run pytest tests/test_grasp_search.py -v
cd /Users/jerry/Code/orca_train
uv run orca-train-state --help
uv run pytest tests/test_train_state.py -v
```

Expected: every help command exits zero and tests pass.

- [ ] **Step 6: Commit Task 10 in each repository**

```bash
cd /Users/jerry/Code/orca_sim
git add src/orca_sim/grasp_search.py tests/test_grasp_search.py
git commit -m "Report grasp acceptance metrics"

cd /Users/jerry/Code/orca_train
git add README.md
git commit -m "Document stable grasp training"
```

---

### Task 11: Run full verification and the local acceptance preflight

**Repositories:** `orca_sim`, `orca_train`

**Files:**
- No source changes expected.
- Generated local smoke artifacts: `/private/tmp/orca-grasp-preflight/`

**Interfaces:**
- Consumes: all preceding tasks.
- Produces: clean full-suite evidence and a short physics preflight report; no long local compute.

- [ ] **Step 1: Run complete test suites from clean processes**

```bash
cd /Users/jerry/Code/orca_sim
uv run pytest tests/

cd /Users/jerry/Code/orca_train
uv run pytest tests/
```

Expected: both suites pass with zero failures.

- [ ] **Step 2: Run static repository checks**

```bash
cd /Users/jerry/Code/orca_sim
git diff --check
git status --short --branch

cd /Users/jerry/Code/orca_train
git diff --check
git status --short --branch
```

Expected: no whitespace errors and only intentional branch-ahead status.

- [ ] **Step 3: Run a bounded local physics preflight**

```bash
mkdir -p /private/tmp/orca-grasp-preflight
cd /Users/jerry/Code/orca_sim
uv run orca-grasp search \
  --population 4 \
  --iterations 1 \
  --search-scenarios 2 \
  --validation-episodes 2 \
  --seed 1 \
  --output /private/tmp/orca-grasp-preflight/grasp_pose.json \
  --report /private/tmp/orca-grasp-preflight/search_report.json \
  --allow-below-threshold
```

Expected: command completes, artifacts are valid JSON, and the report contains
finite metrics. This is a mechanics check, not acceptance evidence.

- [ ] **Step 4: Review verification evidence before remote execution**

Record exact test counts, elapsed times, commit hashes from both repositories,
and the preflight report path in the task handoff. Do not claim stable-grasp
acceptance from the four-candidate local smoke.

---

### Task 12: Run robust search, held-out validation, and curriculum training on Spark

**Environment:** Spark remote host

**Files:**
- Remote isolated root: `/home/jerry/Code/orca_grasp_stability_20260819/`
- Remote search artifacts: `/home/jerry/Code/orca_grasp_stability_20260819/orca_sim/runs/grasp_search_seed1/`
- Remote training artifacts: `/home/jerry/Code/orca_grasp_stability_20260819/orca_train/runs/state_cube_grasp_curriculum_seed1/`
- Local copied artifacts: `/Users/jerry/Code/orca_train/runs/grasp_search_seed1_spark/`

**Interfaces:**
- Consumes: verified commits from Task 11 and Spark CUDA environment.
- Produces: accepted grasp artifact/report locally, then a monitored stage-0-to-stage-4 Spark training run.

- [ ] **Step 1: Create an isolated remote root and sync both verified repositories**

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 spark 'mkdir -p /home/jerry/Code/orca_grasp_stability_20260819'

cd /Users/jerry/Code
rsync -az \
  --exclude .git \
  --exclude .venv \
  --exclude runs \
  orca_sim/ \
  spark:/home/jerry/Code/orca_grasp_stability_20260819/orca_sim/
rsync -az \
  --exclude .git \
  --exclude .venv \
  --exclude runs \
  orca_train/ \
  spark:/home/jerry/Code/orca_grasp_stability_20260819/orca_train/
```

Expected: rsync exits zero for both repositories.

- [ ] **Step 2: Create the remote uv environment and run remote tests**

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 spark 'cd /home/jerry/Code/orca_grasp_stability_20260819/orca_train && uv sync --group dev && uv run pytest tests/ && cd ../orca_sim && ../orca_train/.venv/bin/python -m pytest tests/'
```

Expected: dependency sync and both remote suites pass before search.

- [ ] **Step 3: Launch the full CEM search in a named tmux session**

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 spark 'tmux new-session -d -s orca-grasp-search "cd /home/jerry/Code/orca_grasp_stability_20260819/orca_sim && /home/jerry/Code/orca_grasp_stability_20260819/orca_train/.venv/bin/orca-grasp search --population 128 --iterations 8 --search-scenarios 32 --validation-episodes 100 --required-stability 0.95 --seed 1 --output runs/grasp_search_seed1/grasp_pose.json --report runs/grasp_search_seed1/search_report.json 2>&1 | tee runs/grasp_search_seed1/search.log"'
```

Expected: session `orca-grasp-search` exists and search log begins without import
or artifact errors.

- [ ] **Step 4: Monitor search without starting RL early**

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 spark 'tmux capture-pane -pt orca-grasp-search -S -120'
ssh -o BatchMode=yes -o ConnectTimeout=15 spark 'test -f /home/jerry/Code/orca_grasp_stability_20260819/orca_sim/runs/grasp_search_seed1/grasp_pose.json && test -f /home/jerry/Code/orca_grasp_stability_20260819/orca_sim/runs/grasp_search_seed1/search_report.json'
```

Expected: the artifact existence check succeeds only after search acceptance.
If search exits below 95%, stop here and inspect the preserved best-candidate
report; do not launch training.

- [ ] **Step 5: Run the independent held-out validation**

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 spark 'cd /home/jerry/Code/orca_grasp_stability_20260819/orca_sim && /home/jerry/Code/orca_grasp_stability_20260819/orca_train/.venv/bin/orca-grasp validate --artifact runs/grasp_search_seed1/grasp_pose.json --episodes 100 --seed 20001 --scenario-mode perturbed --required-stability 0.95 --report runs/grasp_search_seed1/validation_report.json'
```

Expected: exit zero, 100 episodes reported, stability at least `0.95`, finite
drift/speed/force metrics, and no hash mismatch.

- [ ] **Step 6: Run nominal 100/100 two-second hold acceptance**

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 spark 'cd /home/jerry/Code/orca_grasp_stability_20260819/orca_sim && /home/jerry/Code/orca_grasp_stability_20260819/orca_train/.venv/bin/orca-grasp validate --artifact runs/grasp_search_seed1/grasp_pose.json --episodes 100 --seed 30001 --scenario-mode nominal --required-stability 1.0 --hold-duration-s 2.0 --report runs/grasp_search_seed1/nominal_validation_report.json'
```

Expected: exit zero with 100 stable episodes, zero drops, and a complete
two-second passive hold in every episode.

- [ ] **Step 7: Copy the accepted grasp artifact and reports locally**

```bash
mkdir -p /Users/jerry/Code/orca_train/runs/grasp_search_seed1_spark
rsync -az \
  spark:/home/jerry/Code/orca_grasp_stability_20260819/orca_sim/runs/grasp_search_seed1/grasp_pose.json \
  spark:/home/jerry/Code/orca_grasp_stability_20260819/orca_sim/runs/grasp_search_seed1/search_report.json \
  spark:/home/jerry/Code/orca_grasp_stability_20260819/orca_sim/runs/grasp_search_seed1/validation_report.json \
  spark:/home/jerry/Code/orca_grasp_stability_20260819/orca_sim/runs/grasp_search_seed1/nominal_validation_report.json \
  /Users/jerry/Code/orca_train/runs/grasp_search_seed1_spark/
```

Expected: all four local files exist and the artifact hash equals all reports.

- [ ] **Step 8: Launch curriculum training only after all acceptance gates pass**

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 spark 'tmux new-session -d -s orca-grasp-train "cd /home/jerry/Code/orca_grasp_stability_20260819/orca_train && .venv/bin/orca-train-state --preset grasp_curriculum --grasp-artifact ../orca_sim/runs/grasp_search_seed1/grasp_pose.json --total-steps 1000000 --eval-every-steps 10000 --eval-episodes 100 --checkpoint-every-steps 50000 --device cuda --seed 1 --output-dir runs/state_cube_grasp_curriculum_seed1 2>&1 | tee runs/state_cube_grasp_curriculum_seed1/train.log"'
```

Expected: config and metrics files appear, initial metrics report curriculum
stage `hold`, and no immediate reset-preparation error occurs.

- [ ] **Step 9: Monitor promotion and enforce stop conditions**

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 spark 'tail -n 100 /home/jerry/Code/orca_grasp_stability_20260819/orca_train/runs/state_cube_grasp_curriculum_seed1/metrics.jsonl'
ssh -o BatchMode=yes -o ConnectTimeout=15 spark 'tmux capture-pane -pt orca-grasp-train -S -120'
```

For every evaluation verify the logged stage, success rate, drop rate, drift,
contacted-finger count, and action saturation. Stop and diagnose if reset
preparation failures recur, NaN/non-finite metrics appear, or drop rate remains
above the stage threshold for three consecutive evaluations. Stage transitions
must appear only after two consecutive passing evaluations.

---

## Final Acceptance Checklist

- [ ] `orca_sim` and `orca_train` full local test suites pass.
- [ ] Remote test suites pass in the isolated Spark root.
- [ ] Nominal grasp has 100/100 two-second zero-action holds without a drop.
- [ ] Independent perturbed validation stability is at least 95%.
- [ ] Policy takeover with zero normalized action produces no joint-target jump.
- [ ] Falling orientation improvement produces no positive total reward.
- [ ] Checkpoint rejects a different grasp artifact hash or curriculum version.
- [ ] Stage 0 reaches at least 99% hold success and at most 1% drops before promotion.
- [ ] Rotation stages promote only at at least 80% success and at most 5% drops for two consecutive evaluations.
- [ ] The accepted grasp artifact and all three reports are copied locally with matching hashes.

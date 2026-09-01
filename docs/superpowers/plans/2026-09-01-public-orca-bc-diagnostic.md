# Public ORCA Behavior-Cloning Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the public `fracapuano/hand-orienting` demonstrations into the production ORCA actor contract, train scratch and warm-start 57-dimensional actors, and compare them with the frozen 20-million-step PPO baseline in the exact Warp/MJX environment.

**Architecture:** A hash-verified downloader reads only LeRobot metadata and Parquet state/action columns. A deterministic converter produces immutable episode files containing canonical actor components, exact 57-dimensional observations, and normalized 16-dimensional relative actions. A JAX behavior-cloning trainer reuses Brax's production PPO policy network and emits a policy bundle that the existing evaluator can execute without a critic.

**Tech Stack:** Python 3.12, `uv`, NumPy, PyArrow, Hugging Face Hub, JAX 0.6.2, Flax 0.11.2, Optax 0.2.6, Brax 0.14.2, MuJoCo 3.6.0, MuJoCo Playground v0.2.0, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-expert-guided-cube-reorientation-design.md`

## Global Constraints

- Execute from a new `feature/expert-guided-cube-reorientation` worktree based on `feature/leap-cube-reorient-ppo` commit `0b791cd76ef4859250fd4676b4fc002e583fbe54`.
- Bring the approved design and this plan into that worktree before changing source.
- Use `uv` for every Python dependency, test, and command.
- Download only `fracapuano/hand-orienting`; do not merge similarly named or rollout datasets.
- Resolve the Hugging Face `main` revision once, then download and record the returned immutable commit SHA and every selected file SHA-256.
- Do not download videos and do not execute remote dataset code.
- Split exactly 20 episodes by episode: 16 training and 4 validation, using one recorded seed.
- Convert 30 Hz to 20 Hz, degrees to radians, remove the wrist, and map 16 named finger joints to `ACTIVE_JOINTS`.
- Actor observations and actions must have shapes `(57,)` and `(16,)`; the BC trainer must not load privileged fields.
- Both BC candidates use policy hidden sizes `(512, 256, 128)` and Huber loss on deterministic action mode.
- This phase is diagnostic. No public-only checkpoint may receive the final `accepted` classification.

## File map

### Create

- `src/orca_train/expert_schema.py` — immutable per-episode NPZ format, manifests, validation, atomic writes, and hashes.
- `src/orca_train/public_orca_data.py` — Hugging Face resolution/download, Parquet validation, joint mapping, resampling, goal relabeling, and conversion CLI.
- `src/orca_train/actor_observation.py` — NumPy construction and noise augmentation for the exact 57-dimensional actor state.
- `src/orca_train/behavior_cloning.py` — production Brax actor construction, BC training, policy bundles, checkpoint initialization, and CLI.
- `src/orca_train/policy_suite.py` — policy-source abstraction and identical-seed six-stage evaluation report.
- `tests/test_expert_schema.py`
- `tests/test_public_orca_data.py`
- `tests/test_actor_observation.py`
- `tests/test_behavior_cloning.py`
- `tests/test_policy_suite.py`
- `tests/test_playground_evaluate.py`

### Modify

- `pyproject.toml` — add an `expert` dependency extra and three console commands.
- `src/orca_train/playground_evaluate.py` — expose the existing rollout evaluator through a generic deterministic policy callback.
- `README.md` — document conversion, BC, and diagnostic evaluation commands.

---

### Task 1: Add the immutable expert episode format

**Files:**
- Create: `src/orca_train/expert_schema.py`
- Create: `tests/test_expert_schema.py`

**Interfaces:**
- Consumes: NumPy arrays and JSON-compatible provenance.
- Produces: `ActorComponents`, `ExpertEpisode`, `EpisodeEntry`, `ExpertDatasetManifest`, `write_episode(path, episode) -> EpisodeEntry`, `finalize_dataset(root, entries, metadata) -> ExpertDatasetManifest`, and `load_episode(path) -> ExpertEpisode`.

- [ ] **Step 1: Write failing shape and finiteness tests**

```python
import numpy as np
import pytest

from orca_train.expert_schema import ActorComponents, ExpertEpisode


def valid_episode(length: int = 3) -> ExpertEpisode:
    components = ActorComponents(
        joint_position=np.zeros((length, 16), np.float32),
        previous_target=np.zeros((length, 16), np.float32),
        palm_position=np.zeros((length, 3), np.float32),
        cube_position=np.zeros((length, 3), np.float32),
        cube_quaternion=np.tile([1, 0, 0, 0], (length, 1)).astype(np.float32),
        goal_quaternion=np.tile([1, 0, 0, 0], (length, 1)).astype(np.float32),
        previous_action=np.zeros((length, 16), np.float32),
    )
    return ExpertEpisode(
        episode_id="public-000000",
        source="fracapuano/hand-orienting",
        stage="public_hindsight",
        observation=np.zeros((length, 57), np.float32),
        next_observation=np.zeros((length, 57), np.float32),
        privileged_observation=None,
        next_privileged_observation=None,
        action=np.zeros((length, 16), np.float32),
        components=components,
        timestamps=np.arange(length, dtype=np.float64) * 0.05,
        extras={"cube_position": np.zeros((length, 3), np.float32)},
        metadata={"source_episode": 0},
    )


def test_episode_contract_accepts_only_57_by_16_finite_arrays() -> None:
    valid_episode().validate()
    broken = valid_episode()
    broken.observation[1, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        broken.validate()


def test_episode_contract_rejects_privileged_width_other_than_131() -> None:
    episode = valid_episode()
    episode.privileged_observation = np.zeros((3, 130), np.float32)
    with pytest.raises(ValueError, match="131"):
        episode.validate()
```

- [ ] **Step 2: Run the new tests and verify the module is missing**

Run: `uv run --extra playground --extra expert pytest tests/test_expert_schema.py -v`

Expected: FAIL during import because `orca_train.expert_schema` does not exist.

- [ ] **Step 3: Implement the schema and validation**

```python
@dataclass
class ActorComponents:
    joint_position: np.ndarray
    previous_target: np.ndarray
    palm_position: np.ndarray
    cube_position: np.ndarray
    cube_quaternion: np.ndarray
    goal_quaternion: np.ndarray
    previous_action: np.ndarray


@dataclass
class ExpertEpisode:
    episode_id: str
    source: str
    stage: str
    observation: np.ndarray
    next_observation: np.ndarray
    privileged_observation: np.ndarray | None
    next_privileged_observation: np.ndarray | None
    action: np.ndarray
    components: ActorComponents
    timestamps: np.ndarray
    extras: dict[str, np.ndarray]
    metadata: dict[str, object]

    def validate(self) -> None:
        length = self.action.shape[0]
        expected = {
            "observation": (length, 57),
            "next_observation": (length, 57),
            "action": (length, 16),
            "timestamps": (length,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        for name in ("privileged_observation", "next_privileged_observation"):
            value = getattr(self, name)
            if value is not None and value.shape != (length, 131):
                raise ValueError(f"{name} must have width 131")
        for name, value in self.extras.items():
            if value.shape[0] != length or not np.all(np.isfinite(value)):
                raise ValueError(f"extra field {name} must be finite and transition-aligned")
```

Validate every component width, strictly increasing timestamps, unit quaternions within `1e-5`, nonempty IDs/source/stage, and JSON-serializable metadata.

- [ ] **Step 4: Write failing atomic round-trip and tamper tests**

```python
from dataclasses import replace

from orca_train.expert_schema import finalize_dataset, load_episode, write_episode


def test_episode_and_manifest_round_trip_with_content_hash(tmp_path) -> None:
    root = tmp_path / "dataset"
    entry = write_episode(root / "episodes/000000.npz", valid_episode())
    manifest = finalize_dataset(root, (entry,), {"revision": "a" * 40})
    assert load_episode(root / entry.relative_path).episode_id == "public-000000"
    assert len(entry.sha256) == len(manifest.content_hash) == 64
    assert not list(root.rglob(".*.tmp"))


def test_loading_tampered_episode_fails(tmp_path) -> None:
    root = tmp_path / "dataset"
    entry = write_episode(root / "episodes/000000.npz", valid_episode())
    path = root / entry.relative_path
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        load_episode(path, expected_sha256=entry.sha256)
```

- [ ] **Step 5: Implement atomic NPZ and manifest IO**

Write episode arrays to an opened temporary binary file with `np.savez`, call `flush` and `os.fsync`, then `os.replace`. Hash the final bytes. Serialize metadata to a sibling JSON entry inside the NPZ as canonical UTF-8 bytes and prefix transition extras with `extra/`. Build the manifest hash from sorted episode entries plus canonical metadata, excluding the `content_hash` field itself.

- [ ] **Step 6: Run schema tests**

Run: `uv run --extra playground --extra expert pytest tests/test_expert_schema.py -v`

Expected: PASS.

- [ ] **Step 7: Commit the schema**

```bash
git add src/orca_train/expert_schema.py tests/test_expert_schema.py
git commit -m "Add immutable expert trajectory schema"
```

---

### Task 2: Add hash-verified public dataset acquisition

**Files:**
- Modify: `pyproject.toml`
- Create: `src/orca_train/public_orca_data.py`
- Create: `tests/test_public_orca_data.py`

**Interfaces:**
- Consumes: Hugging Face repo `fracapuano/hand-orienting` and an output cache directory.
- Produces: `PublicSourceManifest`, `resolve_public_source(api)`, `download_public_source(destination, api)`, and `iter_source_episodes(root)`.

- [ ] **Step 1: Add failing source-selection tests**

```python
from orca_train.public_orca_data import (
    PUBLIC_REPO_ID,
    resolve_public_source,
    selected_source_files,
)


class FakeApi:
    def repo_info(self, **kwargs):
        assert kwargs == {"repo_id": PUBLIC_REPO_ID, "repo_type": "dataset", "revision": "main"}
        return type("Info", (), {"sha": "a" * 40})()


def test_source_revision_is_resolved_to_immutable_sha() -> None:
    source = resolve_public_source(FakeApi())
    assert source.repo_id == "fracapuano/hand-orienting"
    assert source.revision == "a" * 40


def test_selected_files_exclude_videos() -> None:
    selected = selected_source_files()
    assert "meta/*.json" in selected
    assert "data/**/*.parquet" in selected
    assert all("video" not in pattern for pattern in selected)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `uv run pytest tests/test_public_orca_data.py -v`

Expected: FAIL because the module and expert dependencies do not exist.

- [ ] **Step 3: Add the optional dependencies and commands**

Merge the following entries into the existing optional-dependency and script
tables in `pyproject.toml`; do not create duplicate TOML table headers:

```toml
expert = [
    "huggingface-hub>=0.34,<2",
    "pyarrow>=20,<23",
]

[project.scripts]
orca-convert-public-demos = "orca_train.public_orca_data:main"
orca-train-behavior-cloning = "orca_train.behavior_cloning:main"
orca-evaluate-policy-suite = "orca_train.policy_suite:main"
```

Run: `uv lock`

Expected: the lockfile records the new expert dependencies.

Run: `uv sync --extra playground --extra expert --group dev`

Expected: dependency resolution succeeds inside the project environment.

- [ ] **Step 4: Implement immutable revision resolution and restricted download**

```python
PUBLIC_REPO_ID = "fracapuano/hand-orienting"


def selected_source_files() -> tuple[str, ...]:
    return ("meta/*.json", "data/**/*.parquet")


def resolve_public_source(api: HfApi) -> PublicSourceManifest:
    info = api.repo_info(
        repo_id=PUBLIC_REPO_ID,
        repo_type="dataset",
        revision="main",
    )
    if not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise RuntimeError("dataset revision is not an immutable commit SHA")
    return PublicSourceManifest(repo_id=PUBLIC_REPO_ID, revision=info.sha)
```

Call `snapshot_download` with `revision=source.revision`, `allow_patterns=selected_source_files()`, and `repo_type="dataset"`. Hash every selected file and write `source-manifest.json` atomically.

- [ ] **Step 5: Add Parquet contract tests**

Create a two-episode PyArrow fixture containing `timestamp`, `episode_index`, `observation.state`, `cube_pos`, `cube_quat`, and `action`. Assert `iter_source_episodes` groups frames in timestamp order and rejects a missing column, a 16-wide source action, duplicate timestamps, or a non-finite cube quaternion.

- [ ] **Step 6: Implement the strict Parquet reader**

Read only the required columns with `pyarrow.dataset.dataset(source_root / "data", format="parquet").to_table(columns=REQUIRED_COLUMNS)`. Load action and state names from `meta/info.json`, require width 17, group by integer `episode_index`, sort by timestamp, and yield immutable `PublicEpisodeFrames` values. Never infer missing names from array order.

- [ ] **Step 7: Run tests and commit**

Run: `uv run --extra expert pytest tests/test_public_orca_data.py -v`

Expected: PASS.

```bash
git add pyproject.toml uv.lock src/orca_train/public_orca_data.py tests/test_public_orca_data.py
git commit -m "Add verified public ORCA dataset loader"
```

---

### Task 3: Convert source frames into the exact actor contract

**Files:**
- Create: `src/orca_train/actor_observation.py`
- Modify: `src/orca_train/public_orca_data.py`
- Create: `tests/test_actor_observation.py`
- Modify: `tests/test_public_orca_data.py`

**Interfaces:**
- Consumes: `PublicEpisodeFrames`, the active MuJoCo actuator order, actor-noise config, and fixed palm position.
- Produces: `map_source_joints`, `resample_episode_20hz`, `relative_actions`, `build_actor_observation`, `augment_actor_components`, and `convert_public_episode`.

- [ ] **Step 1: Write failing named joint-map tests**

```python
from orca_train.playground_constants import ACTIVE_JOINTS
from orca_train.public_orca_data import map_source_joints


def test_joint_map_drops_wrist_and_maps_thumb_dip_to_thumb_pip() -> None:
    names = (
        "wrist", "pinky_abd", "pinky_mcp", "pinky_pip",
        "ring_abd", "ring_mcp", "ring_pip", "middle_abd",
        "middle_mcp", "middle_pip", "index_abd", "index_mcp",
        "index_pip", "thumb_cmc", "thumb_abd", "thumb_mcp", "thumb_dip",
    )
    indices = map_source_joints(names)
    assert tuple(names[index] for index in indices)[-1] == "thumb_dip"
    assert len(indices) == len(ACTIVE_JOINTS) == 16
    assert 0 not in indices
```

Also assert duplicate names, a missing finger joint, or an extra non-wrist joint raises `ValueError` listing the mismatch.

- [ ] **Step 2: Write failing resampling and action-inversion tests**

```python
def test_resample_30hz_to_20hz_and_invert_relative_targets() -> None:
    source = linear_source_episode(frames=7, hz=30.0)
    converted = resample_episode_20hz(source)
    np.testing.assert_allclose(converted.timestamps, [0.0, 0.05, 0.10, 0.15, 0.20])
    np.testing.assert_allclose(np.linalg.norm(converted.cube_quaternion, axis=1), 1.0)
    actions = relative_actions(converted.target_radians)
    np.testing.assert_allclose(actions, np.diff(converted.target_radians, axis=0) / 0.5)
```

Include a quaternion sign-flip fixture and assert SLERP follows the short arc.

- [ ] **Step 3: Implement mapping, unit conversion, and interpolation**

Normalize source names by removing an optional `right_` prefix and use this exact alias map:

```python
SOURCE_TO_SIM = {
    "pinky_abd": "right_p-abd", "pinky_mcp": "right_p-mcp", "pinky_pip": "right_p-pip",
    "ring_abd": "right_r-abd", "ring_mcp": "right_r-mcp", "ring_pip": "right_r-pip",
    "middle_abd": "right_m-abd", "middle_mcp": "right_m-mcp", "middle_pip": "right_m-pip",
    "index_abd": "right_i-abd", "index_mcp": "right_i-mcp", "index_pip": "right_i-pip",
    "thumb_cmc": "right_t-cmc", "thumb_abd": "right_t-abd",
    "thumb_mcp": "right_t-mcp", "thumb_dip": "right_t-pip",
}
```

Construct the 20 Hz grid with `np.arange(0, floor(duration / 0.05) * 0.05 + 1e-12, 0.05)`. Use linear interpolation for vectors and sign-corrected normalized SLERP for `wxyz` quaternions. Convert action targets with `np.deg2rad` before interpolation. Compute `unclipped = diff(target) / 0.5`, store `clip(unclipped, -1, 1)`, and report total and per-joint saturation fractions in the conversion report so controller mismatch remains visible.

- [ ] **Step 4: Write failing 57-field parity tests**

```python
def test_actor_observation_has_exact_field_slices() -> None:
    components = one_component_sample()
    observation = build_actor_observation(components)
    assert observation.shape == (57,)
    np.testing.assert_allclose(observation[:16], components.joint_position)
    np.testing.assert_allclose(observation[16:32], components.joint_position - components.previous_target)
    np.testing.assert_allclose(observation[32:35], components.palm_position - components.cube_position)
    np.testing.assert_allclose(observation[41:57], components.previous_action)
```

Build one CPU `OrcaCubeReorient` state with actor noise disabled and assert the NumPy observation equals `state.obs["state"]` to `1e-6`, including the rotation-matrix `ravel()[3:]` slice.

- [ ] **Step 5: Implement canonical observation construction and exact noise augmentation**

Compute the quaternion difference as `cube_quaternion * inverse(goal_quaternion)`, convert it to a rotation matrix, and take flattened elements 3 through 8. `augment_actor_components` applies the configured uniform joint noise, uniform cube-position noise, and normalized Gaussian cube-quaternion noise before calling `build_actor_observation`. It must consume an explicit NumPy generator and never mutate stored components.

- [ ] **Step 6: Implement hindsight conversion**

Use the final finite resampled cube quaternion as every sample's goal. Drop the first frame because it has no preceding target, and use frames `1` through `N-2` as current states so the terminal hindsight-goal frame can supply the final next observation. Populate current and next canonical observations, `previous_target`, `previous_action`, fixed palm position, timestamps, cube pose extras, source metadata, and actions. Validate active target ranges against `OrcaCubeReorient.active_ctrl_lower` and `active_ctrl_upper`.

- [ ] **Step 7: Run focused tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_actor_observation.py tests/test_public_orca_data.py -v`

Expected: PASS.

```bash
git add src/orca_train/actor_observation.py src/orca_train/public_orca_data.py tests/test_actor_observation.py tests/test_public_orca_data.py
git commit -m "Convert public demos to Leap actor contract"
```

---

### Task 4: Add public conversion CLI and deterministic episode split

**Files:**
- Modify: `src/orca_train/public_orca_data.py`
- Modify: `tests/test_public_orca_data.py`

**Interfaces:**
- Consumes: grasp artifact, validation report, source cache, output directory, and split seed.
- Produces: `convert_public_dataset(source_root, output_root, grasp_artifact, validation_report, split_seed) -> ExpertDatasetManifest` and `orca-convert-public-demos`.

- [ ] **Step 1: Write a failing end-to-end conversion test**

Use the two-episode Parquet fixture, mocked immutable source revision, and a CPU test grasp. Assert the CLI writes `episodes/*.npz`, `manifest.json`, and `conversion-report.json`; all episodes have shape `(T, 57)/(T, 16)`; and rerunning into a populated destination raises `FileExistsError`.

- [ ] **Step 2: Write a failing 16/4 split test**

```python
from orca_train.public_orca_data import split_episode_ids


def test_public_split_is_deterministic_and_has_no_episode_leakage() -> None:
    episode_ids = tuple(f"public-{index:06d}" for index in range(20))
    first = split_episode_ids(episode_ids, seed=20260901)
    second = split_episode_ids(episode_ids, seed=20260901)
    assert first == second
    assert len(first.train) == 16
    assert len(first.validation) == 4
    assert set(first.train).isdisjoint(first.validation)
```

- [ ] **Step 3: Implement conversion orchestration**

Resolve/download the source, instantiate the CPU environment only to obtain the hash-bound palm and actuator contract, convert episodes in numeric order, write immutable entries, and finalize the manifest with:

```python
metadata = {
    "source_repo": PUBLIC_REPO_ID,
    "source_revision": source.revision,
    "source_file_hashes": source.file_hashes,
    "conversion_version": 1,
    "goal_relabel": "final_finite_cube_quaternion",
    "source_hz": 30.0,
    "target_hz": 20.0,
    "split_seed": split_seed,
    "train_episode_ids": split.train,
    "validation_episode_ids": split.validation,
    "model_signature": environment.validated_grasp.artifact.settled_state.model_signature,
    "active_joints": ACTIVE_JOINTS,
}
```

- [ ] **Step 4: Run conversion tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_expert_schema.py tests/test_public_orca_data.py tests/test_actor_observation.py -v`

Expected: PASS.

```bash
git add src/orca_train/public_orca_data.py tests/test_public_orca_data.py
git commit -m "Add reproducible public demo conversion"
```

---

### Task 5: Train production-network behavior-cloning actors

**Files:**
- Create: `src/orca_train/behavior_cloning.py`
- Create: `tests/test_behavior_cloning.py`

**Interfaces:**
- Consumes: an `ExpertDatasetManifest`, split name, optional PPO checkpoint, and `BehaviorCloningConfig`.
- Produces: `BehaviorCloningBundle`, `make_bc_networks`, `state_normalizer`, `train_behavior_cloning(dataset_root, output_dir, initialization, parent_checkpoint, config)`, `load_behavior_cloning_bundle(path)`, and `make_bc_policy(bundle)`.

- [ ] **Step 1: Write failing architecture and privileged-access tests**

```python
def test_bc_uses_production_actor_and_never_requests_privileged_state() -> None:
    networks = make_bc_networks()
    params = networks.policy_network.init(jax.random.PRNGKey(0))
    normalizer = state_normalizer(np.zeros((4, 57), np.float32))
    logits = networks.policy_network.apply(
        normalizer,
        params,
        {"state": jnp.zeros((4, 57), dtype=jnp.float32)},
    )
    assert logits.shape[-1] == 32
```

Pass an observation mapping whose `privileged_state` property raises on access and assert a training step succeeds.

- [ ] **Step 2: Write failing loss/update tests**

Create a four-sample dataset where actions are a fixed linear function of observation. Assert one JIT update reduces Huber loss, changes policy parameters, produces zero direct gradient for the tanh-normal scale-logit half on a fixed hidden activation, and is deterministic for the same seed.

- [ ] **Step 3: Implement production actor and deterministic action loss**

Build networks with:

```python
network_config = orca_ppo_config().network_factory
networks = ppo_networks.make_ppo_networks(
    observation_size={"state": 57, "privileged_state": 131},
    action_size=16,
    **network_config,
)
```

Apply the policy network and obtain deterministic actions with `networks.parametric_action_distribution.mode(logits)`. Optimize `optax.huber_loss(prediction, action, delta=0.1).mean()` using Adam at `3e-4`. Do not add a scale or entropy term: the deterministic mode depends on the mean-logit half, so the scale-logit half receives no direct BC loss while the shared production trunk remains trainable.

- [ ] **Step 4: Implement scratch and warm-start initialization**

For scratch, initialize the policy with the run seed. For warm start, call `ppo_checkpoint.load(checkpoint)` and take element 1 as policy parameters. Validate its `ppo_network_config.json`, action schema, observation schema, active joint order, and 57-dimensional policy key before use. Recompute the public training-split state normalizer for both candidates; do not load the PPO critic.

- [ ] **Step 5: Write failing bundle round-trip tests**

Save `normalizer_params`, `policy_params`, network config, dataset hash, parent checkpoint hash, split seed, epoch, best validation loss, and source revisions. Reload the bundle and require bitwise-equal deterministic actions on a fixed batch. Tampering with `metadata.json` or parameter bytes must raise a hash error.

- [ ] **Step 6: Implement episode-balanced training and early stopping**

Sample an episode uniformly, then a transition within that episode. Apply actor noise from stored raw components with a per-batch PRNG. Train at most 200 epochs, define one epoch as one pass over the number of training transitions, evaluate the fixed-noise validation split each epoch, stop after 20 epochs without improvement, and retain only the lowest validation-loss bundle.

- [ ] **Step 7: Run tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_behavior_cloning.py -v`

Expected: PASS.

```bash
git add src/orca_train/behavior_cloning.py tests/test_behavior_cloning.py
git commit -m "Add Brax-compatible behavior cloning"
```

---

### Task 6: Evaluate BC bundles and PPO checkpoints on identical stages

**Files:**
- Create: `src/orca_train/policy_suite.py`
- Modify: `src/orca_train/playground_evaluate.py`
- Create: `tests/test_policy_suite.py`
- Create: `tests/test_playground_evaluate.py`

**Interfaces:**
- Consumes: `PolicySource(kind, path)`, six curriculum stages, fixed seed lists, grasp inputs, and domain randomization.
- Produces: `STAGE_NAMES`, `StageResult`, `PolicySuiteReport`, `make_evaluation_wrapper`, `stage_step_record`, `evaluate_policy_suite(policy_source, grasp_artifact, validation_report, episodes_per_stage, seed_root)`, and `public_bc_is_useful(report, baseline) -> bool`.

- [ ] **Step 1: Extract a generic deterministic-policy rollout seam**

Add this parameter to the existing stage rollout core rather than duplicating environment stepping:

```python
PolicyFn = Callable[[Mapping[str, jax.Array], jax.Array], tuple[jax.Array, Mapping[str, Any]]]


def rollout_policy_stage(
    policy: PolicyFn,
    *,
    environment: OrcaCubeReorient,
    episodes: int,
    seed: int,
    steps: int,
    domain_randomization: bool,
) -> StageResult:
    if episodes <= 0 or steps <= 0:
        raise ValueError("episodes and steps must be positive")
    batched_environment = make_evaluation_wrapper(
        environment,
        episodes=episodes,
        seed=seed,
        domain_randomization=domain_randomization,
    )
    reset_keys = jp.stack(
        [jax.random.PRNGKey(seed + index) for index in range(episodes)]
    )
    initial_state = jax.jit(batched_environment.reset)(reset_keys)

    def scan_step(carry, unused):
        del unused
        state, policy_key = carry
        policy_key, action_key = jax.random.split(policy_key)
        action, policy_extras = policy(state.obs, action_key)
        next_state = batched_environment.step(state, action)
        record = stage_step_record(
            environment,
            state,
            next_state,
            action,
            policy_extras,
        )
        return (next_state, policy_key), record

    (_, _), rollout = jax.jit(
        lambda state, key: jax.lax.scan(
            scan_step,
            (state, key),
            (),
            length=steps,
        )
    )(initial_state, jax.random.PRNGKey(seed + 2_000_000))
    return StageResult.from_rollout(rollout, seed=seed, episodes=episodes)
```

Keep `evaluate_checkpoint` and `evaluate_hold_checkpoint` as compatibility wrappers around this seam.

- [ ] **Step 2: Write failing source-loader tests**

Assert `load_policy_source` loads a Brax checkpoint through `load_checkpoint_policy`, loads a BC bundle through `make_bc_policy`, and rejects a source whose model/grasp/action/observation fingerprints differ from the requested evaluation environment.

- [ ] **Step 3: Write failing six-stage and usefulness tests**

```python
def test_public_bc_usefulness_requires_turn_gain_without_drop_regression() -> None:
    baseline = suite(success={"turn_30": 0.0}, aggregate_drop=0.20)
    useful = suite(success={"turn_30": 0.01}, aggregate_drop=0.20)
    unsafe = suite(success={"turn_30": 0.10}, aggregate_drop=0.21)
    assert public_bc_is_useful(useful, baseline)
    assert not public_bc_is_useful(unsafe, baseline)
```

Require stage order `hold`, `turn_10_15`, `turn_30`, `turn_60`, `turn_90`, `full_so3`; 256 seeds per stage for the diagnostic; and exact reuse of the same seed list across sources.

- [ ] **Step 4: Implement reports and atomic IO**

Report per-stage success, drop, mean/median orientation error, return, time to first success, wrist drift, non-finite count, and per-seed records. Include policy, environment, grasp, reset, seed-list, and report hashes. Set `classification="diagnostic"` for every public-only bundle.

- [ ] **Step 5: Run focused tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_policy_suite.py tests/test_playground_evaluate.py -v`

Expected: PASS.

```bash
git add src/orca_train/policy_suite.py src/orca_train/playground_evaluate.py tests/test_policy_suite.py tests/test_playground_evaluate.py
git commit -m "Compare policies on identical reorientation stages"
```

---

### Task 7: Add the diagnostic CLIs, documentation, and execution gate

**Files:**
- Modify: `src/orca_train/public_orca_data.py`
- Modify: `src/orca_train/behavior_cloning.py`
- Modify: `src/orca_train/policy_suite.py`
- Modify: `README.md`
- Modify: `tests/test_public_orca_data.py`
- Modify: `tests/test_behavior_cloning.py`
- Modify: `tests/test_policy_suite.py`

**Interfaces:**
- Consumes: local/remote source directories, the frozen baseline checkpoint, grasp/reset inputs, and explicit output directories.
- Produces: converted public dataset, two BC bundles, three identical-seed suite reports, and `public-bc-selection.json`.

- [ ] **Step 1: Add CLI parsing tests**

Assert:

- conversion defaults to repo `fracapuano/hand-orienting`, split seed `20260901`, and target rate `20`;
- BC requires `--dataset`, `--output-dir`, and `--initialization {scratch,ppo}`; `ppo` also requires `--parent-checkpoint`;
- suite evaluation defaults to 256 episodes per stage and requires explicit `--policy-kind {ppo,bc}`.

- [ ] **Step 2: Implement fail-fast CLIs**

Each CLI writes its resolved config before work begins, refuses a nonempty output directory, emits JSON on success, and emits a failure manifest before re-raising any data/schema/non-finite error. `public-bc-selection.json` names `scratch`, `warm_start`, or `none` and records the exact usefulness-rule inputs.

- [ ] **Step 3: Document exact commands**

Add these commands with real paths replaced only by shell variables defined immediately above them:

```bash
uv run --extra playground --extra expert orca-convert-public-demos \
  --grasp-artifact runs/inputs/grasp.json \
  --validation-report runs/inputs/grasp-validation.json \
  --output-dir runs/public_hand_orienting_v1 \
  --split-seed 20260901

uv run --extra playground --extra expert orca-train-behavior-cloning \
  --dataset runs/public_hand_orienting_v1 \
  --initialization scratch \
  --output-dir runs/public_bc_scratch_seed1 \
  --seed 1

uv run --extra playground --extra expert orca-train-behavior-cloning \
  --dataset runs/public_hand_orienting_v1 \
  --initialization ppo \
  --parent-checkpoint runs/orca_leap_reorient_20m/checkpoints/20000000 \
  --output-dir runs/public_bc_warm_seed1 \
  --seed 1
```

Document the equivalent three suite evaluations and the rule that no public-only result is accepted.

- [ ] **Step 4: Run the complete local test gate**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_expert_schema.py tests/test_public_orca_data.py tests/test_actor_observation.py tests/test_behavior_cloning.py tests/test_policy_suite.py tests/test_playground_evaluate.py -v`

Expected: PASS.

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests -q`

Expected: all repository tests pass.

- [ ] **Step 5: Run the public conversion and CPU BC smoke**

Run the documented conversion command, then both BC commands with `--max-epochs 2`. Verify 20 input episodes, 16/4 split, finite loss, distinct output directories, bundle round-trip action parity, and no privileged-loader access.

- [ ] **Step 6: Commit the completed diagnostic path**

```bash
git add README.md pyproject.toml uv.lock src/orca_train/expert_schema.py src/orca_train/public_orca_data.py src/orca_train/actor_observation.py src/orca_train/behavior_cloning.py src/orca_train/policy_suite.py src/orca_train/playground_evaluate.py tests/test_expert_schema.py tests/test_public_orca_data.py tests/test_actor_observation.py tests/test_behavior_cloning.py tests/test_policy_suite.py tests/test_playground_evaluate.py
git commit -m "Document public ORCA BC diagnostic"
```

- [ ] **Step 7: Run exact-environment diagnostic evaluation on Spark**

Sync the committed `orca_train` and matching `orca_sim` snapshots into the versioned directory `/data/home/scv7454/run/orca_expert_guided_20260901` without `--delete`. Install with `uv sync --extra playground --extra expert --group dev`. Require `jax.default_backend()` to report `gpu`, then evaluate the frozen baseline and both BC bundles with 256 identical seeds per stage. Copy the three reports and `public-bc-selection.json` back to the local `runs/public_bc_diagnostic_20260901/` directory and verify SHA-256 hashes.

Expected: the selection file chooses exactly one of `scratch`, `warm_start`, or `none`; training does not proceed to public-only PPO.

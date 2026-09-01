# Expert-Guided Student PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pretrain a deployable 57-dimensional student from verified demonstrations, continue it with a freshly initialized 131-dimensional asymmetric PPO critic, and deliver an accepted policy only after the six-stage 6,000-episode gate passes.

**Architecture:** Build stage-balanced train/validation/test splits from the immutable exact-environment dataset, optionally retain the useful public BC initializer, and train the production actor in mixed then exact-only BC phases. Convert the BC bundle into Brax `restore_params` so PPO restores only actor/normalizer state and initializes a new privileged critic. Evaluate the frozen baseline, public BC, exact BC, and BC-plus-PPO policies with identical seeds, then bind final export to the accepted suite report.

**Tech Stack:** Python 3.12, `uv`, NumPy, JAX 0.6.2, Flax 0.11.2, Optax 0.2.6, Brax 0.14.2, Orbax Checkpoint 0.11.31, MuJoCo 3.6.0, MuJoCo Warp/MJX, MuJoCo Playground v0.2.0, pytest, Spark GB10 GPU.

**Spec:** `docs/superpowers/specs/2026-09-01-expert-guided-cube-reorientation-design.md`

## Global Constraints

- Complete both `2026-09-01-public-orca-bc-diagnostic.md` and `2026-09-01-privileged-teacher-expert-data.md` first.
- Use the same `feature/expert-guided-cube-reorientation` worktree and `uv` environment.
- The deployable actor consumes exactly 57 dimensions; only the PPO value network consumes 131 dimensions.
- Student BC uses Huber loss on the normalized 16-dimensional deterministic action mode and never reads privileged arrays in its forward or loss path.
- If public BC passed its diagnostic, train first with 80% exact teacher transitions and 20% public transitions, then exact teacher transitions only. Otherwise use exact data only.
- Split exact trajectories per stage into 80 train, 10 validation, and 10 test episodes when the minimum count is 100; additional episodes are assigned 80/10/10 by deterministic round-robin after seeded shuffle.
- The default path contains no PPO BC auxiliary loss and no DAgger. At most one DAgger round is allowed only by the explicit trigger in Task 3.
- Student PPO uses balanced resets, the unchanged Leap hyperparameters, a 2-million-step gate, and a 20-million-step ceiling.
- Final evaluation uses 1,000 held-out seeds per stage and identical seeds for every compared policy.
- Export only a policy with hold success `>=0.99`, hold drop `<=0.01`, each turning success `>=0.80`, full-SO3 drop `<=0.05`, no non-finite episodes, and wrist drift `<=1e-8`.

## File map

### Create

- `src/orca_train/student_data.py` — deterministic exact/public splits, stage/episode-balanced sampling, source mixture, and DAgger revision metadata.
- `src/orca_train/student_bc.py` — mixed and exact-only BC orchestration, full observation normalizer construction, and student bundle CLI.
- `src/orca_train/student_run.py` — BC-to-Brax restore conversion, student PPO gate/continuation, and CLI.
- `src/orca_train/dagger_collect.py` — single-round trigger and privileged teacher corrections on student-visited states.
- `src/orca_train/final_evaluate.py` — four-candidate, 6,000-episode comparison and acceptance report.
- `tests/test_student_data.py`
- `tests/test_student_bc.py`
- `tests/test_student_run.py`
- `tests/test_dagger_collect.py`
- `tests/test_final_evaluate.py`

### Modify

- `src/orca_train/behavior_cloning.py` — reusable epoch trainer and full 57/131 normalizer bundle.
- `src/orca_train/playground_run.py` — `restore_params` and fresh-value restore support.
- `src/orca_train/playground_export.py` — bind export to final suite acceptance.
- `src/orca_train/policy_suite.py` — final 1,000-seed metrics and percentile fields.
- `pyproject.toml` — student BC, PPO, DAgger, and final evaluation commands.
- `README.md` — complete student training, gate, comparison, and export workflow.
- `tests/test_behavior_cloning.py`
- `tests/test_playground_run.py`
- `tests/test_playground_export.py`
- `tests/test_policy_suite.py`

---

### Task 1: Build deterministic stage-balanced student splits

**Files:**
- Create: `src/orca_train/student_data.py`
- Create: `tests/test_student_data.py`

**Interfaces:**
- Consumes: expert and optional public manifests plus split seed.
- Produces: `DatasetSplit`, `StudentDataPlan`, `build_student_data_plan`, and `sample_student_batch`.

- [ ] **Step 1: Write failing exact split tests**

```python
from orca_train.student_data import build_student_data_plan


def test_exact_split_is_80_10_10_per_stage_without_seed_leakage() -> None:
    manifest = expert_manifest(episodes_per_stage=100)
    plan = build_student_data_plan(manifest, public=None, split_seed=20260901)
    for stage in STAGE_NAMES:
        assert len(plan.exact.train[stage]) == 80
        assert len(plan.exact.validation[stage]) == 10
        assert len(plan.exact.test[stage]) == 10
        assert set(plan.exact.train[stage]).isdisjoint(plan.exact.validation[stage])
        assert set(plan.exact.train[stage]).isdisjoint(plan.exact.test[stage])
        assert set(plan.exact.validation[stage]).isdisjoint(plan.exact.test[stage])
```

Assert reset seeds are globally disjoint across splits and rerunning with the same seed produces byte-identical JSON.

- [ ] **Step 2: Write failing public selection tests**

Pass `public-bc-selection.json` values `scratch`, `warm_start`, and `none`. Require the matching public bundle hash for the first two and no public samples for `none`. Reject a selection file whose dataset or evaluation report hash differs.

- [ ] **Step 3: Implement split planning and hash binding**

Shuffle episode IDs independently per stage using `np.random.default_rng(split_seed + stage_index)`. Assign the first 80%, next 10%, and final 10%, with minimum counts 80/10/10. Record all episode IDs, reset seeds, source hashes, selected public bundle, teacher checkpoint, teacher gate, and collection report in `student-data-plan.json` and content-hash it.

- [ ] **Step 4: Write failing sampler tests**

Sample 60,000 exact transitions and assert stage frequencies are within `0.01` of `1/6`. Within one stage, assert episode frequencies are uniform within statistical tolerance even when episode lengths differ by 10x. In mixed mode, assert source frequency is `0.80 +/- 0.01` exact and `0.20 +/- 0.01` public.

- [ ] **Step 5: Implement hierarchical sampling**

For exact samples, draw stage uniformly, episode uniformly within stage, then transition uniformly within episode. For public samples, draw episode uniformly, then transition. Choose the source first with the configured probability. Return canonical raw components, canonical observation, action, and source metadata; privileged state is returned only when `include_privileged=True` is explicitly requested by normalizer/audit code.

- [ ] **Step 6: Run tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_student_data.py -v`

Expected: PASS.

```bash
git add src/orca_train/student_data.py tests/test_student_data.py
git commit -m "Add balanced student demonstration splits"
```

---

### Task 2: Train mixed then exact-only student BC

**Files:**
- Create: `src/orca_train/student_bc.py`
- Modify: `src/orca_train/behavior_cloning.py`
- Create: `tests/test_student_bc.py`
- Modify: `tests/test_behavior_cloning.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `StudentDataPlan`, selected public initializer or scratch seed, and BC configuration.
- Produces: `StudentBCBundle`, `train_student_bc`, `make_full_observation_normalizer`, and `orca-train-student-bc`.

- [ ] **Step 1: Write failing two-phase schedule tests**

```python
def test_student_bc_uses_mixed_then_exact_only_batches() -> None:
    calls = []
    result = train_student_bc(
        data_plan=fake_plan(public_selected=True),
        train_epoch=lambda phase, source_weights, **_: calls.append((phase, source_weights)) or finite_epoch(),
        config=tiny_config(),
    )
    assert calls[0] == ("mixed", {"exact": 0.8, "public": 0.2})
    assert calls[-1] == ("exact_only", {"exact": 1.0, "public": 0.0})
    assert result.best_phase == "exact_only"
```

With `public_selected=False`, assert both phases have exact weight 1.0.

- [ ] **Step 2: Write failing split-isolation and normalization tests**

Assert training statistics use only exact teacher training episodes, never public, validation, or test episodes. Build the normalizer with keys `state` width 57 and `privileged_state` width 131. Verify the BC actor accesses only the selected `state` normalizer while the stored privileged normalizer is finite and available for the future critic.

- [ ] **Step 3: Implement full observation normalizer**

Initialize Brax running statistics for:

```python
observation_shapes = {
    "state": jax.ShapeDtypeStruct((57,), jnp.float32),
    "privileged_state": jax.ShapeDtypeStruct((131,), jnp.float32),
}
normalizer = running_statistics.init_state(observation_shapes)
```

Update the two-key tree in one call stream using only exact teacher training transitions: augmented actor observations under `state` and their aligned privileged observations under `privileged_state`. This preserves Brax's single shared running-statistics count. Public rows still train the actor but do not affect PPO normalization. Assert the final count, mean, and standard deviation are finite and the shape tree exactly matches PPO environment observations.

- [ ] **Step 4: Implement two-phase early stopping**

Reuse the JIT update from `behavior_cloning.py`. Mixed phase: at most 200 epochs, patience 20, validation on exact validation episodes. Exact-only phase: restore the best mixed parameters, train at most 100 epochs, patience 20, validation on the same exact split. Retain the lowest exact-validation Huber bundle. Apply online actor noise with fixed validation seeds and no privileged input to the actor/loss.

- [ ] **Step 5: Add bundle provenance and round-trip tests**

Require dataset plan hash, expert manifest hash, optional public bundle/report hashes, teacher checkpoint/gate hashes, phase histories, normalizer hash, source revisions, and policy-action parity after reload. Reject a bundle if `policy_obs_key != "state"`, observation width is not 57, or action width is not 16.

- [ ] **Step 6: Run tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_behavior_cloning.py tests/test_student_bc.py -v`

Expected: PASS.

```bash
git add pyproject.toml src/orca_train/behavior_cloning.py src/orca_train/student_bc.py tests/test_behavior_cloning.py tests/test_student_bc.py
git commit -m "Train exact-environment BC student"
```

---

### Task 3: Add one explicitly triggered DAgger recovery round

**Files:**
- Create: `src/orca_train/dagger_collect.py`
- Create: `tests/test_dagger_collect.py`
- Modify: `src/orca_train/student_bc.py`
- Modify: `src/orca_train/student_data.py`

**Interfaces:**
- Consumes: student BC report, frozen teacher report/checkpoint, exact validation baseline, and student-visited states.
- Produces: `dagger_is_required`, one immutable `dagger_corrections` dataset revision, and retrained BC bundle.

- [ ] **Step 1: Write failing trigger tests**

```python
def test_dagger_trigger_requires_offline_gain_and_closed_loop_collapse() -> None:
    assert dagger_is_required(
        initial_validation_loss=0.20,
        final_validation_loss=0.17,
        student_mean_turn_success=0.39,
        teacher_mean_turn_success=0.80,
        previous_dagger_rounds=0,
    )
    assert not dagger_is_required(0.20, 0.19, 0.39, 0.80, 0)
    assert not dagger_is_required(0.20, 0.17, 0.40, 0.80, 0)
    assert not dagger_is_required(0.20, 0.17, 0.10, 0.80, 1)
```

The exact trigger is validation loss improvement of at least 10% and student mean turning success strictly below 50% of teacher mean turning success, with zero previous DAgger rounds.

- [ ] **Step 2: Write failing correction-data tests**

Roll out a fake student, query a fake teacher on the same privileged states, and assert stored actions are teacher actions while stored actor observations are the student's visited 57-dimensional states. Label the dataset `dagger_corrections`, not `expert`, and record both policy hashes.

- [ ] **Step 3: Implement a single correction round**

Use fixed 128-seed batches per turning stage, stop after 10,000 corrections or complete seed exhaustion, and exclude hold. The student acts in the environment; the teacher action is recorded before every student step. Reject non-finite states/actions and wrist drift. Finalize a new immutable data-plan revision that retains the original expert manifest and adds the correction manifest.

- [ ] **Step 4: Retrain from the pre-DAgger BC best bundle**

Sample exact successful data and DAgger corrections 1:1 inside the exact source portion, preserve any already-approved 20% public portion only during the mixed phase, then finish exact/correction-only. Do not run a second DAgger trigger.

- [ ] **Step 5: Run tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_dagger_collect.py tests/test_student_data.py tests/test_student_bc.py -v`

Expected: PASS.

```bash
git add src/orca_train/dagger_collect.py src/orca_train/student_bc.py src/orca_train/student_data.py tests/test_dagger_collect.py tests/test_student_data.py tests/test_student_bc.py
git commit -m "Add bounded DAgger recovery round"
```

---

### Task 4: Restore only the BC actor into asymmetric PPO

**Files:**
- Modify: `src/orca_train/playground_run.py`
- Create: `src/orca_train/student_run.py`
- Modify: `tests/test_playground_run.py`
- Create: `tests/test_student_run.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `StudentBCBundle`, balanced environment inputs, and student budget.
- Produces: `bc_restore_params(bundle)`, generalized `build_train_fn(ppo_config, seed, checkpoint_dir, resume_checkpoint, restore_params, restore_value_fn, progress_fn)`, `train_student_ppo`, and `orca-train-expert-student`.

- [ ] **Step 1: Write failing Brax restore-parameter tests**

```python
def test_bc_restore_tuple_contains_normalizer_and_policy_but_no_value() -> None:
    restore = bc_restore_params(fake_student_bundle())
    assert restore[0] is fake_student_bundle().normalizer_params
    assert restore[1] is fake_student_bundle().policy_params
    assert restore[2] is None
```

Reject a bundle whose network config differs from `orca_ppo_config().network_factory`, except that it contains no trained value network.

- [ ] **Step 2: Extend `build_train_fn` without changing checkpoint resume**

Add keyword parameters:

```python
restore_params: Any | None = None,
restore_value_fn: bool = True,
```

Pass both to `ppo.train`. Reject calls that supply both `resume_checkpoint` and `restore_params`. Existing checkpoint resume keeps `restore_value_fn=True`; BC initialization uses `restore_params=(normalizer, policy, None)` and `restore_value_fn=False`.

- [ ] **Step 3: Prove the value network is fresh**

Run `ppo.train` with `num_timesteps=0` twice, once from scratch and once with BC restore parameters, using the same seed. Assert restored policy equals BC policy, restored normalizer equals BC normalizer, and restored value parameters equal the scratch seeded value parameters rather than any parent checkpoint value.

- [ ] **Step 4: Write failing student manifest tests**

Require role `student`, policy key `state`, value key `privileged_state`, balanced curriculum, initialization kind `expert_bc`, BC bundle/data hashes, and allowed 2M/20M budget. Require 20M checkpoint continuation to preserve all initialization lineage from the 2M manifest.

- [ ] **Step 5: Implement the student PPO wrapper**

`train_student_ppo` loads and verifies the BC bundle, constructs the unchanged student `orca_ppo_config`, sets the requested allowed budget, and calls generalized `train_playground` with balanced mode and actor-only restore parameters. A 20M continuation resumes the 2M PPO checkpoint normally; it does not reload BC directly.

- [ ] **Step 6: Add privileged-access negative test**

Replace the policy observation mapping with a guard whose `privileged_state` access raises and assert inference/training policy application succeeds. Replace the value mapping with a guard whose `state` access raises and assert value application succeeds. This test must exercise the actual production network factory.

- [ ] **Step 7: Run tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_playground_run.py tests/test_student_run.py -v`

Expected: PASS.

```bash
git add pyproject.toml src/orca_train/playground_run.py src/orca_train/student_run.py tests/test_playground_run.py tests/test_student_run.py
git commit -m "Warm start asymmetric PPO from BC actor"
```

---

### Task 5: Add student continuation gates and checkpoint selection

**Files:**
- Modify: `src/orca_train/student_run.py`
- Modify: `src/orca_train/policy_suite.py`
- Modify: `tests/test_student_run.py`
- Modify: `tests/test_policy_suite.py`

**Interfaces:**
- Consumes: BC closed-loop report, initial/2M/20M PPO suite reports.
- Produces: `student_smoke_allows_continuation`, `rank_student_reports`, and `student-gate.json`.

- [ ] **Step 1: Write failing continuation tests**

Require at least one turning-stage success to exceed the BC actor's corresponding result and aggregate drop to be no more than `BC + 0.01`. Require finite losses/gradients/parameters, all six reset stages represented, no abnormal termination, and wrist drift `<=1e-8`. Assert each condition fails independently.

- [ ] **Step 2: Implement student ranking**

Rank candidates by descending mean six-stage success, descending full-SO3 success, ascending aggregate drop, then checkpoint hash. Record every candidate, rule input, and tie-break field. Never overwrite or delete the BC bundle or original 20M baseline.

- [ ] **Step 3: Add gate CLI**

Add `orca-gate-expert-student --bc-report runs/exact-bc-suite.json --candidate-report runs/student-2m-suite.json --output runs/student-smoke-gate.json`. It writes a gate file on pass or fail and returns nonzero on failure so the remote supervisor cannot accidentally launch the continuation.

- [ ] **Step 4: Run tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_student_run.py tests/test_policy_suite.py -v`

Expected: PASS.

```bash
git add pyproject.toml src/orca_train/student_run.py src/orca_train/policy_suite.py tests/test_student_run.py tests/test_policy_suite.py
git commit -m "Gate expert-guided student continuation"
```

---

### Task 6: Add final four-candidate evaluation and acceptance

**Files:**
- Create: `src/orca_train/final_evaluate.py`
- Create: `tests/test_final_evaluate.py`
- Modify: `src/orca_train/policy_suite.py`
- Modify: `tests/test_policy_suite.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: frozen original PPO, optional selected public BC, exact BC, final PPO, 1,000-seed lists, and shared environment inputs.
- Produces: `FinalComparisonReport`, `accept_student_suite`, `evaluate_final_comparison`, and `orca-evaluate-expert-student-final`.

- [ ] **Step 1: Write failing acceptance-boundary tests**

```python
def test_final_acceptance_exact_boundaries_pass() -> None:
    report = final_student_report(
        hold_success=0.99,
        hold_drop=0.01,
        turning_success=[0.80] * 5,
        full_drop=0.05,
        max_wrist=1e-8,
        nonfinite=0,
        episodes_per_stage=1000,
    )
    assert accept_student_suite(report)
```

Parametrize one failure at `0.989`, `0.011`, `0.799`, `0.051`, `1.1e-8`, one non-finite episode, and 999 episodes per stage.

- [ ] **Step 2: Add percentile and time-to-success metrics**

For every stage report mean, median, p90, p95, and p99 final orientation error; mean episode return; mean/median time to first success; goals completed; drop rate and mean drop step; wrist drift; non-finite and abnormal termination counts; and all per-seed outcomes. Unit-test percentiles on a known five-value array.

- [ ] **Step 3: Enforce identical seed/environment bindings**

Build six immutable lists of 1,000 seeds starting from a recorded root seed with disjoint ranges. Require all four candidates to use the same seed-list hash, grasp/reset/domain-randomization/model hashes, episode lengths, and stage definitions. If public selection is `none`, retain a `public_bc` entry with status `not_selected` rather than changing report shape.

- [ ] **Step 4: Implement final comparison and report hashing**

Evaluate candidates in fixed order `baseline_ppo_20m`, `public_bc`, `exact_teacher_bc`, `expert_bc_ppo`. The accepted flag is computed only from `expert_bc_ppo`. Hash every source artifact and the canonical report. Refuse a preexisting destination or mismatched candidate provenance.

- [ ] **Step 5: Run tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_policy_suite.py tests/test_final_evaluate.py -v`

Expected: PASS.

```bash
git add pyproject.toml src/orca_train/policy_suite.py src/orca_train/final_evaluate.py tests/test_policy_suite.py tests/test_final_evaluate.py
git commit -m "Evaluate final student on six stages"
```

---

### Task 7: Bind policy-only export to the accepted final suite

**Files:**
- Modify: `src/orca_train/playground_export.py`
- Modify: `tests/test_playground_export.py`

**Interfaces:**
- Consumes: accepted `FinalComparisonReport` and matching final PPO checkpoint.
- Produces: a 57-dimensional `PolicyBundle` with normalization, action/joint contracts, source hashes, and deterministic parity proof.

- [ ] **Step 1: Write failing export-gate tests**

Assert export rejects an unaccepted report, a report for another checkpoint, fewer than 1,000 episodes per stage, a privileged policy key, a mismatched seed/environment/grasp hash, or any missing compared candidate. Assert it accepts the exact boundary report from Task 6.

- [ ] **Step 2: Update export binding**

Replace separate hold/full report requirements with one final comparison report. Store its SHA-256 and the selected student's per-stage metrics. Preserve policy parameters and `state` normalization only; do not serialize teacher policy, value parameters, optimizer state, or privileged input as deployable inputs.

- [ ] **Step 3: Verify checkpoint/bundle action parity**

Build a fixed finite batch of 32 observations with shape `(32, 57)`. Require deterministic checkpoint and exported bundle actions to match with `rtol=1e-6, atol=1e-6`, have shape `(32, 16)`, remain in `[-1, 1]`, and reproduce after bundle reload.

- [ ] **Step 4: Run tests and commit**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests/test_playground_export.py tests/test_final_evaluate.py -v`

Expected: PASS.

```bash
git add src/orca_train/playground_export.py tests/test_playground_export.py
git commit -m "Bind student export to final acceptance"
```

---

### Task 8: Execute student BC, PPO, final evaluation, and delivery

**Files:**
- Modify: `README.md`
- Modify: `tests/test_student_bc.py`
- Modify: `tests/test_student_run.py`
- Modify: `tests/test_final_evaluate.py`

**Interfaces:**
- Consumes: verified public diagnostic, expert dataset, teacher artifacts, original baseline, and committed source snapshots.
- Produces: student data plan, BC bundle, optional DAgger revision, 2M/optional-20M PPO checkpoints, final report, accepted export, videos, and reproduction manifest.

- [ ] **Step 1: Add and test exact student CLI contracts**

Require explicit hashes/paths for public selection, expert manifest, teacher gate, BC output, PPO output, final report, and export. Defaults are split seed `20260901`, student seed 1, 80/20 mixed source weights, balanced resets, 2M smoke, 20M ceiling, and 1,000 final episodes per stage.

- [ ] **Step 2: Document the end-to-end commands**

Document creation of `student-data-plan.json`, student BC, closed-loop BC evaluation, optional DAgger gate/collection, student 2M PPO, continuation gate, 20M resume, final comparison, export, and video rendering. Every command uses paths under the versioned remote run root and writes a unique output directory.

- [ ] **Step 3: Run the full local regression gate**

Run: `JAX_PLATFORMS=cpu uv run --extra playground --extra expert pytest tests -q`

Expected: all tests pass.

- [ ] **Step 4: Run a local CPU BC integration smoke**

Use synthetic six-stage expert episodes and a two-epoch configuration. Require finite mixed/exact loss, deterministic bundle reload, a 57-only guarded actor, a full 57/131 normalizer tree, and one `num_timesteps=0` actor-only PPO restore proving the critic is fresh.

- [ ] **Step 5: Sync and verify the committed snapshots on Spark**

Update `/data/home/scv7454/run/orca_expert_guided_20260901` without `--delete`, refresh `source-revisions.json`, run `uv sync --extra playground --extra expert --group dev`, execute the full tests, assert the JAX GPU backend, and verify every input artifact hash before training.

- [ ] **Step 6: Train and evaluate the exact BC student**

Build the deterministic 80/10/10-per-stage data plan. Train mixed then exact-only BC, run the 256-seed-per-stage closed-loop suite, and evaluate the DAgger trigger. If triggered, perform exactly one correction round and retrain; otherwise record `dagger_status="not_required"`.

- [ ] **Step 7: Run the 2-million-step student PPO gate**

Start a named persistent session with 8,192 environments, seed 1, balanced resets, exact Leap hyperparameters, actor-only BC restore, fresh critic, and output `runs/expert_student_seed1_2m`. Require finite metrics, stage coverage, wrist invariance, and restorable checkpoints. Evaluate candidates with the same 256 seeds as BC and write `student-smoke-gate.json`.

- [ ] **Step 8: Continue only after the gate**

If the gate fails, stop and preserve evidence. If it passes but the final acceptance gate is not yet met, resume the selected 2M PPO checkpoint into `runs/expert_student_seed1_20m` with the 20,000,000 ceiling. Preserve all parent hashes and do not reload BC directly during resume.

- [ ] **Step 9: Run the final identical-seed comparison**

Evaluate baseline PPO, selected-or-none public BC, exact teacher BC, and final expert BC+PPO on 1,000 held-out seeds for each of six stages. Write the 6,000-episode-per-candidate comparison with per-seed outcomes and apply the exact acceptance boundary.

- [ ] **Step 10: Export only on acceptance**

If accepted, export the 57-dimensional actor, verify checkpoint/bundle parity, render representative hold and full-SO3 videos from the accepted checkpoint, and write reproduction commands. If not accepted, deliver reports/checkpoints but do not create an accepted policy bundle.

- [ ] **Step 11: Copy and hash-verify final artifacts locally**

Copy all data plans, BC/DAgger artifacts, student manifests/checkpoints, metrics, logs, final report, export, normalization state, and videos to local `runs/expert_guided_20260901/`. Compare local hashes to remote manifests before claiming completion.

- [ ] **Step 12: Commit documentation**

```bash
git add README.md tests/test_student_bc.py tests/test_student_run.py tests/test_final_evaluate.py
git commit -m "Document expert-guided student training"
```

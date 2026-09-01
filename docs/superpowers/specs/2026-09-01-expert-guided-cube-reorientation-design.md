# Expert-Guided ORCA Cube Reorientation Design

**Date:** 2026-09-01

**Status:** Approved

**Repositories:** `orca_train`, `orca_sim`

**Baseline design:** `2026-08-20-leap-cube-reorient-ppo-design.md`

## Summary

The existing ORCA `LeapCubeReorient` adaptation can keep the wrist fixed and
run the intended asymmetric PPO stack, but the latest 20-million-step run did
not learn reliable cube reorientation. This design replaces the blocked
single-path curriculum with a stage-balanced reset distribution and adds two
sources of imitation guidance:

1. a small public ORCA demonstration dataset for a cheap behavior-cloning
   diagnostic; and
2. successful trajectories generated in the exact Warp/MJX environment by a
   privileged PPO teacher.

The deployable student remains a 57-observation, 16-action policy. Its PPO
critic remains asymmetric and consumes the 131-dimensional privileged state.
The privileged teacher is never exported for deployment.

## Relationship to the baseline design

This design retains the following baseline requirements:

- MuJoCo Playground `v0.2.0` and the upstream `LeapCubeReorient` Brax PPO
  implementation;
- a fixed ORCA wrist and 16 active finger actions;
- relative target actions with scale `0.5`;
- the Leap reward, observation noise, domain randomization, termination,
  timing, and `(512, 256, 128)` policy/value networks;
- a 57-dimensional deployable actor observation;
- a 131-dimensional privileged observation containing the fifth fingertip;
- validated stable-grasp resets and hash-checked artifacts; and
- exact-environment evaluation on held-out seeds.

This design supersedes these parts of the baseline:

- the 200-million-step budget;
- promotion through a single global curriculum stage;
- the requirement that only an unmodified parity run may be accepted; and
- random-policy-only initialization.

Teacher and student PPO runs use 2 million steps as a mandatory smoke gate and
may continue to at most 20 million steps when the smoke metrics are finite and
show useful stage-wise learning. Every intentional departure from upstream
Leap is recorded in the run manifest.

## Evidence and root cause

The latest 20-million-step warm-start run completed without NaNs, out-of-memory
errors, or wrist motion. Its fixed-seed evaluation nevertheless reported:

- hold success `0%`, drop `0%`, maximum orientation error `0.594 rad`;
- full-SO(3) success `1%`, drop `21%`, mean orientation error `2.051 rad`; and
- average curriculum stage about `1.29`, between the `10-15 deg` and `30 deg`
  stages.

The result rules out a crashed optimizer as the primary explanation. The
training distribution remained concentrated on early goals and supplied too
few successful rotation transitions to bootstrap the harder task. More steps
on the same reset process are therefore not the first remedy. The corrective
design changes the data distribution and adds successful action labels while
keeping the deployable observation and action contracts intact.

## Goals

1. Measure whether existing public ORCA demonstrations provide a useful actor
   initialization despite their schema and timing mismatch.
2. Train a privileged teacher in the exact ORCA Warp/MJX environment without
   exposing privileged inputs to the deployable actor.
3. Generate a balanced, hash-verified dataset containing only successful
   exact-environment trajectory segments.
4. Pretrain the 57-dimensional student on demonstrations, then continue with
   the approved 57/131 asymmetric PPO setup.
5. Evaluate the original PPO baseline, public-data BC, teacher-data BC, and
   BC-plus-PPO student on identical held-out seeds.

## Non-goals

- The teacher is not a deployable policy.
- Public videos are not used as visual observations.
- Hardware demonstrations and real-hand fine-tuning are outside this phase.
- The initial implementation does not add a permanent BC auxiliary term to
  PPO, a new reward, a recurrent policy, or a new policy architecture.
- The public ACT checkpoint is not silently treated as compatible. It may be
  evaluated later through a separately validated adapter, but it is not needed
  for the first direct-BC diagnostic or the exact-environment teacher path.
- Failed teacher episodes are preserved for diagnostics but are not included
  in the first expert dataset.

## Public assets and selection

The initial direct-BC diagnostic uses only
[`fracapuano/hand-orienting`](https://huggingface.co/datasets/fracapuano/hand-orienting):

- Apache-2.0 license;
- 20 episodes and 6,834 frames;
- 30 Hz sampling;
- 17-dimensional ORCA joint state including the wrist;
- cube position and quaternion;
- 17-dimensional absolute joint targets in degrees; and
- wrist and top-down videos, which are ignored here.

The following discovered assets are not merged into the initial dataset:

- `fracapuano/orcahand-orienting`, because it appears to duplicate the same 20
  episodes;
- `fracapuano/hand-orienting-rollouts`, because its ten episodes have no
  machine-readable success labels and must first pass exact-environment
  validation;
- `fracapuano/inhand-orientation`, because it represents a separate 11-episode
  collection; and
- `fracapuano/inhand-real-all-success`, because it is real-hardware data and
  therefore outside the initial simulation-only seed.

Every downloaded asset is pinned by repository revision and file SHA-256. The
loader reads the declared columns directly and does not execute unreviewed
remote code.

## Public trajectory conversion

The public source cannot be consumed directly because it differs from the
student environment in frequency, wrist control, action semantics, and goal
labels. Conversion is a deterministic, versioned preprocessing step.

### Timing and units

- Resample from 30 Hz to the environment's 20 Hz control grid using source
  timestamps.
- Linearly interpolate joint positions, cube positions, and absolute targets.
- Interpolate cube orientations with sign-corrected quaternion SLERP, then
  normalize and canonicalize them.
- Convert targets from degrees to radians before any action calculation.
- Reject episodes with non-monotonic timestamps, missing frames, non-finite
  values, or quaternions below the configured normalization tolerance.

### Joint and action mapping

- Resolve the 17 source joints by name; never rely on column position alone.
- Remove the wrist and reorder the remaining 16 joints into the simulator's
  active-actuator order.
- Preserve the repository's documented `thumb_dip` to simulated-thumb-PIP
  semantic mapping.
- Validate all converted targets against the MuJoCo actuator ranges.
- Reconstruct the normalized relative action from consecutive absolute
  targets using the environment's own target-update inverse:

  ```text
  action[t] = clip((target[t] - target[t-1]) / 0.5, -1, 1)
  ```

  The first resampled frame has no preceding target and is not used as an
  action-labelled sample.

### Goal relabeling and 57-dimensional observations

The public dataset has no goal quaternion. Each frame is relabelled with an
achieved future cube pose from the same episode. The initial diagnostic uses
the final finite cube quaternion of the episode as the goal and excludes the
terminal frame itself. This produces a causal demonstration of motion toward
an achieved pose without claiming knowledge of the original operator's goal.

The converter reconstructs the exact 57 fields in the actor schema:

- 16 active joint angles;
- 16 joint-angle-minus-previous-target values;
- 3 palm-to-cube position values;
- 6 cube-to-relabelled-goal rotation values; and
- the previous reconstructed 16-dimensional action.

The fixed palm position is derived from the same hash-verified MJCF model used
for evaluation; it is not estimated from the public video. Conversion fails if
that model hash or the fixed-wrist configuration differs from the manifest.

Observation noise is not synthesized into the stored canonical sample. The BC
dataloader applies the same configured actor noise online, with validation
using a fixed noise seed. Every sample stores its source episode, frame time,
goal-relabel rule, joint-map version, and conversion version.

## Direct behavior-cloning diagnostic

The 20 public episodes are split by episode, never by frame:

- 16 training episodes;
- 4 validation episodes; and
- a fixed split seed recorded in the manifest.

Two actors with the production `(512, 256, 128)` architecture are trained:

1. **Public BC from scratch.** All policy parameters are initialized normally.
2. **Baseline fine-tune.** The actor from the completed 20-million-step PPO run
   is loaded and fine-tuned on the same converted data.

BC optimizes only the policy action mean with Huber loss on the normalized
16-dimensional action. It does not train a critic and never reads the
131-dimensional privileged state. The production Brax tanh-normal actor emits
mean and scale logits from one network; BC supervises only the distribution
mode, so scale logits receive no direct loss. The full actor is retained for
subsequent PPO rather than replacing it with a BC-specific output head.

Both candidates are evaluated in the exact Warp/MJX environment, along with
the frozen original PPO baseline, on hold, `10-15 deg`, `30 deg`, `60 deg`,
`90 deg`, and full-SO(3) tasks. This diagnostic does not by itself produce an
accepted model. The public-data branch remains eligible as a student
initializer only if it improves at least one non-hold stage success rate over
the original baseline without increasing aggregate drop rate. Ties are broken
by lower validation Huber loss, then lower full-SO(3) orientation error.

If neither BC candidate meets this usefulness rule, public data remains an
archived negative result and the student starts from exact-environment teacher
data instead.

## Privileged teacher

The teacher is a separate PPO policy whose actor and critic both consume the
existing 131-dimensional `privileged_state`. It outputs the same normalized
16-dimensional relative action as the student. The wrist remains fixed and the
teacher uses the same Leap reward, timing, termination, network widths, domain
randomization, and action transformation.

The teacher is trained independently of the public BC actor. This prevents the
public dataset's approximate goals, CPU physics, and absolute-target controller
from becoming hidden assumptions in the expert policy.

### Stage-balanced resets

At every episode reset, select one of these task stages with probability
`1/6`:

1. hold;
2. random-axis `10-15 deg` turn;
3. random-axis `30 deg` turn;
4. random-axis `60 deg` turn;
5. random-axis `90 deg` turn; or
6. full SO(3).

The selected stage remains fixed for the episode. After a bounded-stage goal
is achieved, a new goal from the same stage is sampled. Full SO(3) retains the
upstream Leap success-triggered goal-update behavior. Hold evaluation requires
two seconds of stability and does not count a momentary orientation match as a
success.

This sampler replaces global stage promotion for teacher and student training.
Stage-wise metrics are still checkpointed and reported, but no stage can
prevent harder resets from entering the replay-free on-policy batch.

### Teacher training gate

- Run a 2-million-step smoke phase first.
- Require a GPU backend, finite observations, actions, rewards, losses,
  gradients, parameters, and checkpoints.
- Require exact wrist invariance and nonzero representation of all six reset
  stages.
- Continue to at most 20 million steps only if at least one turning stage
  improves over its initial fixed-seed evaluation while aggregate drop rate is
  no more than one percentage point above its initial value.
- Select checkpoints by the mean of the six stage success rates, with
  full-SO(3) success and then drop rate as tie breakers.

The teacher is eligible to generate the balanced expert dataset only after it
achieves at least `99%` hold success and at least `80%` success on each of the
five turning stages on the intermediate held-out evaluation set.

## Exact-environment expert trajectories

Trajectory collection runs in the same Warp/MJX environment, with the same
domain-randomization distribution, observation code, action code, and fixed
wrist as student PPO.

### Acceptance of a trajectory

For a turning task, a stored trajectory segment starts at reset and ends at
the first environment-defined goal success. For hold, it ends after two
seconds of successful stability. A segment is accepted only when:

- the unchanged environment success predicate is satisfied;
- no drop or non-finite state occurred before success;
- the wrist's maximum absolute deviation is at most `1e-8`;
- every action lies in `[-1, 1]`; and
- its reset seed, stage, domain-randomization parameters, model fingerprint,
  teacher checkpoint, and code revisions are known.

Rejected and failed rollouts are logged separately with a reason code. They do
not enter the first expert training split.

### Balance and schema

Collect at least 100 accepted trajectory segments for each of the six stages,
for at least 600 total. Collection uses independent seeds from teacher training
and evaluation. Full-SO(3) goals are stratified into 24 cells: three geodesic
angle bands (`[0, pi/3)`, `[pi/3, 2*pi/3)`, and `[2*pi/3, pi]`) crossed with
the eight rotation-axis octants. At least four accepted segments are required
from every cell, preventing a high count from one narrow goal region.

Each transition stores:

- canonical noise-free 57-dimensional student observation;
- 131-dimensional privileged state for audit and possible teacher replay;
- normalized 16-dimensional action;
- next student and privileged observations;
- cube pose and linear/angular velocity;
- goal quaternion and orientation error;
- reward, success, drop, termination, and truncation fields;
- curriculum stage and within-episode goal index;
- reset seed and domain-randomization parameters; and
- all schema, environment, model, grasp, and checkpoint fingerprints.

The dataset writer uses episode-level atomic files plus a manifest. A completed
dataset is immutable and identified by a content hash. Student loaders expose
only the 57-dimensional observation and 16-dimensional action by default;
access to privileged columns requires an explicit teacher/audit mode.
Student BC applies the configured actor observation noise online to both public
and exact-environment canonical observations, using fixed seeds for validation.

## Student imitation learning

The public diagnostic selects an initializer, not a final policy. Student BC
then runs in two phases:

1. train with batches containing `80%` exact-environment teacher transitions
   and `20%` converted public transitions; and
2. finish with exact-environment teacher transitions only.

If public data failed its usefulness rule, both phases use exact-environment
teacher data and the selected public initializer is discarded. Sampling is
balanced by curriculum stage and then by episode so long trajectories cannot
dominate the loss.

Student BC retains the 57-input `(512, 256, 128)` actor and uses Huber loss on
the normalized action mean. Train/validation/test splits are made by episode
and reset seed. Observation-normalization statistics are estimated from the BC
training split only and stored with the actor.

The BC checkpoint is evaluated closed-loop in the exact environment. If its
offline validation improves while its closed-loop success collapses through
compounding error, one DAgger round is permitted: roll out the current student,
query the frozen privileged teacher on those visited states, append the teacher
corrections as a new immutable dataset revision, and retrain. DAgger is a
triggered recovery step, not part of the default path.

## Asymmetric student PPO

The PPO student restores:

- the BC actor parameters;
- the BC actor observation-normalization statistics; and
- the complete production actor, including its scale-logit outputs.

The 131-dimensional critic and its optimizer state are freshly initialized.
The actor can access only `state` with 57 dimensions; the value network alone
uses `privileged_state` with 131 dimensions. A preflight access test fails if
the policy network is connected to any privileged slice.

Student PPO uses the same stage-balanced reset sampler as the teacher. The
upstream Leap PPO hyperparameters remain unchanged except for the explicit
2-million-step gate and 20-million-step ceiling. No BC or teacher loss is added
to PPO initially. If later evidence shows catastrophic imitation forgetting,
that change requires a new diagnostic manifest and cannot be silently enabled.

Student PPO continuation follows the same finite-metric, wrist-invariance, and
stage-coverage smoke gates as the teacher. Checkpoints are ranked by mean
stage-wise success, then full-SO(3) success, then lower aggregate drop rate.

## Evaluation and acceptance

All candidates use the same grasp artifact, environment fingerprint, domain
randomization, and fixed seed lists. The compared candidates are:

1. frozen original 20-million-step PPO baseline;
2. selected public-data BC actor;
3. exact-teacher-data BC student; and
4. exact-teacher-data BC plus asymmetric PPO student.

Intermediate evaluation uses 256 fixed seeds per stage. Final evaluation uses
1,000 fixed held-out seeds for each of hold, `10-15 deg`, `30 deg`, `60 deg`,
`90 deg`, and full SO(3), for 6,000 episodes per candidate.

The final student is accepted only when:

- hold success is at least `99%`;
- hold drop rate is at most `1%`;
- success is at least `80%` on every turning stage, including full SO(3);
- full-SO(3) drop rate is at most `5%`;
- wrist maximum absolute deviation is at most `1e-8`;
- no episode contains NaN, Inf, or an unclassified abnormal termination; and
- the restorable checkpoint and policy-only export produce matching
  deterministic actions on a fixed observation batch.

The report also includes mean, median, and percentile orientation error;
episode return; time to first success; goals completed; drop timing; and
per-seed outcomes. A result from a cherry-picked seed or a different reset
distribution cannot receive the accepted label.

## Artifacts and provenance

Each run or dataset manifest records:

- `orca_train` and `orca_sim` revisions;
- Playground, Brax, JAX, MuJoCo, CUDA, and driver versions;
- effective PPO and environment configs;
- public dataset revision, license, and file hashes;
- conversion, joint-map, action, observation, and trajectory schema versions;
- grasp artifact, MJCF model, and actuator-order hashes;
- random seeds and domain-randomization configuration;
- source and parent checkpoint hashes; and
- diagnostic, teacher, student, or accepted classification.

Final delivery contains the original baseline report, public conversion report,
BC comparison report, privileged teacher checkpoint, immutable successful
trajectory dataset, student BC checkpoint, best restorable PPO checkpoint,
57-dimensional policy-only bundle, normalization state, final 6,000-episode
report, representative videos, and reproduction commands.

## Error handling

- A missing or mismatched source column, joint name, actuator range, schema,
  hash, or checkpoint stops the relevant stage before training.
- Quaternion normalization, interpolation, and action inversion are checked
  against finite numerical tolerances; invalid episodes are quarantined rather
  than partially converted.
- Public and exact-environment samples cannot be mixed without an explicit
  source tag and conversion version.
- A 57/131 observation-shape mismatch or privileged-to-policy connection fails
  before JIT compilation.
- NaN/Inf in state, observation, action, reward, loss, gradient, parameter, or
  metric terminates the run and writes a failure manifest.
- A failed 2-million-step gate does not automatically consume the remaining
  budget.
- Failure to meet teacher gates prevents the dataset from being labelled
  expert; failure to meet final gates prevents the student from being labelled
  accepted.
- Remote interruption resumes only from a manifest-compatible, hash-verified
  checkpoint.

## Test strategy

Implementation follows test-driven development.

### Conversion tests

- named 17-to-16 joint mapping and wrist removal;
- degrees-to-radians conversion and actuator-range validation;
- 30-to-20 Hz timestamps and interpolation;
- quaternion sign correction, SLERP, normalization, and canonicalization;
- absolute-target to relative-action inversion and clipping;
- future-achieved goal relabeling;
- exact 57-dimensional field order; and
- episode-level split with no frame leakage.

### Dataset and BC tests

- public source revision and hash verification;
- immutable manifest and content-hash round trip;
- privileged fields hidden from the default student loader;
- stage- and episode-balanced sampling;
- Huber loss and policy-mean-only updates;
- scratch and warm-start checkpoint compatibility; and
- deterministic normalization and fixed-seed validation.

### Environment and teacher tests

- exact 57/131 observation shapes and field order;
- teacher policy/value both receiving 131 dimensions;
- student policy receiving 57 and value receiving 131 dimensions;
- fixed wrist and 16-dimensional action mapping;
- uniform six-stage reset frequencies within statistical tolerance;
- stage remains fixed across within-episode goal refreshes;
- unchanged Leap reward, termination, randomization, and action semantics;
- successful-trajectory gate and rejection reason codes; and
- checkpoint restore preserves sampler and normalization state.

### Integration and remote gates

- CPU conversion/BC smoke test;
- Warp/MJX batched reset and step test;
- finite gradient update for teacher, BC student, and PPO student;
- policy privileged-access negative test;
- exact-environment closed-loop evaluation on all six stages;
- interrupted-run checkpoint resume; and
- deterministic checkpoint-versus-export action parity.

## Execution sequence

1. Implement and test the public data downloader, converter, and immutable
   dataset schema.
2. Run the two public BC diagnostics and exact-environment comparison.
3. Implement and test the stage-balanced reset sampler and privileged teacher
   entry point.
4. Run the 2-million-step teacher gate; continue only when its gate passes.
5. Evaluate the teacher and collect at least 600 accepted balanced trajectories.
6. Train and evaluate the 57-dimensional BC student.
7. Run the 2-million-step asymmetric-PPO student gate; continue only when its
   gate passes.
8. Perform the final identical-seed comparison and export only an accepted
   student.

No Spark training is started until local tests, source synchronization, remote
dependency checks, model compilation, and a shortened GPU rollout pass.

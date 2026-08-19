# Sim-to-Real Initial Grasp Stability Design

**Date:** 2026-08-19  
**Status:** Approved for implementation planning  
**Repositories:** `orca_sim`, `orca_train`

## Problem statement

The current v2 cube-orientation task does not begin from a grasp. The cube is
placed on an open palm and the task policy receives control immediately. A
diagnostic rollout using the exact 150k training configuration showed:

- the reset state has only one cube contact, with the palm/carpals;
- a zero-action policy drops the cube after six 80 ms control steps, about
  0.48 seconds;
- the 150k policy reaches a transient best orientation error of 3.496 degrees
  at step seven, when cube height is 0.1114 m and angular speed is
  12.60 rad/s;
- the policy loses all contact and drops the cube on the next step.

The policy therefore learned to rotate the cube while it was falling rather
than to perform stable in-hand manipulation. A larger drop penalty alone does
not correct the invalid initial state or prevent orientation progress from
being earned during a fall.

## Goals

1. Start every rotation episode from a physically simulated, stable grasp.
2. Place the cube at the center of the palm at initialization, expressed in
   the hand mount frame rather than as a fixed world coordinate.
3. Use only motions and sensing assumptions that can be reproduced on the
   physical ORCA Hand.
4. Prevent falling or high-speed cube motion from producing useful orientation
   reward.
5. Introduce rotation through a stability-first curriculum instead of starting
   with a 90-degree target.
6. Search and train at scale on Spark; use the local machine only for unit
   tests and short physics checks.

## Non-goals

- Autonomous pickup is not part of this task. A human or fixture may place the
  cube at the known palm-center location before the scripted closing motion.
- No weld, mocap attachment, invisible support geometry, gravity change, or
  other simulation-only retention mechanism is allowed.
- Contact measurements used by simulation reset validation or training rewards
  are not added to the deployed policy observation.

## Architecture

The environment reset is divided into two explicit phases:

1. **Grasp preparation:** reset physics, place the cube at the named palm
   center, execute a smooth closing trajectory, hold, and validate stability.
2. **Rotation task:** after the grasp passes validation, clear task counters,
   sample the orientation target, initialize the policy target from measured
   joint positions, and expose the first observation.

Preparation time does not count toward the RL episode and does not contribute
reward or replay transitions. A failed preparation is retried up to a bounded
limit and is never handed to the policy.

### Palm-centered cube placement

The v2 hand model will expose a named site on the palm/carpals body. The cube
center is placed at this site's world position after the hand is set to the
open pose and forward kinematics are evaluated. A small configurable offset is
expressed in the site's local frame so calibration does not reintroduce world
coordinate constants. The default offset is the calibrated palm center. The
initial cube orientation remains an explicit task setting.

The same site-to-palm definition will be documented for physical placement.
The physical workflow is: place the cube at the marked palm center, command the
same closing trajectory, verify stability, then enable the rotation policy.

### Transferable closing trajectory

The open pose and grasp pose are named joint-angle configurations. The wrist is
held fixed during grasp preparation. Finger commands interpolate from the open
pose to the grasp pose with a smoothstep trajectory over a configurable default
of 0.6 seconds. Position, velocity, joint-range, and actuator-force limits must
remain consistent with the physical controller.

After closing, the controller holds the final pose for a configurable default
of 0.5 seconds. The final measured joint positions, not the nominal command,
become the initial target of the residual rotation controller. This avoids a
control discontinuity at policy takeover.

### Stability gate

The last 0.25 seconds of the hold window are evaluated. A reset is stable only
if all configured conditions pass:

- cube height remains above the safety threshold;
- cube displacement relative to the palm is at most 5 mm over the window;
- relative linear speed is at most 0.05 m/s;
- relative angular speed is at most 0.5 rad/s;
- the cube contacts the palm and at least two distinct fingers;
- contact normals provide opposing support rather than contacts exclusively on
  one side;
- contact force and actuator effort remain below transfer-calibrated limits.

Thresholds are configuration values and will be calibrated from the robust
grasp search. Simulation may use contact geometry for reset qualification. The
physical equivalent uses the tracked cube pose and motor current/effort; the RL
policy receives neither signal as a new observation.

## Robust grasp-pose search

The search fixes `right_wrist` and optimizes the 16 finger joints. Candidates
must remain inside joint ROM with a configurable safety margin and must be
reachable by the transferable closing trajectory.

A dependency-light cross-entropy method (CEM) performs the search:

1. sample candidate finger poses from a bounded distribution;
2. run the complete place-close-hold sequence for every candidate;
3. reject candidates that drop the cube, violate limits, or fail minimum
   contact topology;
4. score valid candidates using hold duration, palm-relative drift and speed,
   opposing contact support, peak contact force, actuator effort, and distance
   from joint limits;
5. update the sampling distribution from robust elite candidates;
6. repeat until the budget or convergence limit is reached.

Each candidate is evaluated across perturbations of cube placement,
orientation, mass, friction, and actuator response. Ranking uses a lower-tail
robust score rather than the mean so a pose cannot win by overfitting one
nominal state. Search and validation use disjoint deterministic seed sets.

The selected pose must achieve at least 95% stability on the held-out
validation set. Search produces a versioned grasp-pose artifact and a report
containing joint targets, trajectory settings, scenario statistics, failure
reasons, and a content hash. Training configs and checkpoints record this hash.

The full CEM search runs on Spark. Local execution is limited to deterministic
smoke cases and short physics validation.

## Rotation control and safe exploration

The rotation policy continues to output bounded residual joint actions, but
initial exploration changes:

- remove the first 5,000 full-range uniformly random actions;
- seed replay with zero actions and small residual perturbations around the
  validated grasp;
- constrain targets to a safety envelope around the grasp pose during early
  curriculum stages;
- widen the envelope only as stability and rotation performance improve;
- penalize rapid action changes and persistent normalized-action saturation.

The safety envelope is a training constraint, not an unmodeled physical aid.
It is applied through the same joint target limits that can run on the real
controller.

## Grasp-aware reward

The reward separates rotation progress from grasp health:

- angular-error reduction is multiplied by a grasp-quality gate derived from
  palm-relative cube position and motion;
- high-speed or out-of-workspace cube motion receives no positive orientation
  progress;
- palm-relative drift, linear speed, angular speed, action rate, and action
  saturation receive continuous penalties;
- target completion still requires orientation tolerance, sufficient height,
  low linear and angular speed, and the success hold duration;
- drop is terminal and its penalty is greater than the discounted upper bound
  of all non-terminal positive reward available in one episode.

This makes a transient orientation improvement followed by a drop strictly
worse than preserving the grasp. Contact-derived grasp quality may be used as
privileged training reward information but is not part of the policy input.

## Curriculum

Training advances through the following checkpointed stages:

| Stage | Task | Promotion criterion |
|---|---|---|
| 0 | Hold the initial orientation for 2 seconds | hold rate at least 99%, drop rate at most 1% |
| 1 | Rotate 10-15 degrees | success at least 80%, drop at most 5% |
| 2 | Rotate 30 degrees | success at least 80%, drop at most 5% |
| 3 | Rotate 60 degrees | success at least 80%, drop at most 5% |
| 4 | Rotate 90 degrees | terminal training stage |

Promotion is based on fixed held-out evaluation seeds and must pass consecutive
evaluations to reduce sampling noise. The current stage and all stage settings
are saved in the checkpoint so resume is deterministic.

## Components and repository boundaries

### `orca_sim`

- add the named palm grasp-center site to the v2 model;
- add grasp-pose configuration and validation;
- implement place-close-hold reset preparation;
- expose structured stability metrics and bounded failure reasons;
- add the CEM search entry point and validation report generation.

The generic environment remains usable without grasp preparation through an
explicit configuration boundary. State cube training enables the validated
grasp reset by default.

### `orca_train`

- add grasp-reset and grasp-artifact fields to state training configuration;
- add safe replay seeding and the stage-dependent action envelope;
- add grasp-aware reward configuration and curriculum state;
- record stability time, cube drift, contacted-finger count, action saturation,
  drop reason, and current curriculum stage in metrics;
- embed the grasp artifact hash and curriculum state in checkpoints and reject
  incompatible resume attempts.

## Data flow

1. Load the validated grasp artifact and verify its hash.
2. Reset the hand to the open pose.
3. Evaluate forward kinematics and place the cube at the palm-center site.
4. Run the closing and hold trajectory under normal MuJoCo physics.
5. Evaluate the rolling stability window.
6. Retry a bounded number of times on failure; otherwise report a structured
   reset error.
7. On success, initialize policy control from measured joint state and start
   the current curriculum task.
8. Compute grasp-aware reward and detailed metrics at every policy step.
9. Evaluate promotion criteria on the held-out seed set and checkpoint stage
   transitions.

## Error handling

- Reset retries have a fixed maximum and never loop indefinitely.
- A failed reset reports the failed conditions, best observed values, seed,
  grasp artifact hash, and perturbation parameters.
- NaN or non-finite state, force, or reward terminates the attempt with an
  explicit failure reason.
- Training cannot start if the grasp artifact is missing, fails schema or hash
  validation, or does not include its held-out validation report.
- Checkpoint resume fails clearly when grasp hash, observation/action schema,
  or curriculum definition differs.
- If no search candidate reaches the 95% validation threshold, the pipeline
  stops before RL training and preserves the best candidates for diagnosis.

## Test and acceptance strategy

Implementation follows test-driven development.

### Unit tests

- palm-site local-to-world cube placement;
- smooth, monotonic, ROM-safe closing trajectory;
- stability-window thresholds and contact classification;
- bounded reset retries and structured failure reporting;
- deterministic CEM sampling and robust candidate ranking;
- reward gives no positive falling-orientation exploit;
- drop penalty dominates all possible non-terminal positive reward;
- safe exploration envelope and action-rate metrics;
- curriculum promotion, persistence, and incompatible-resume checks.

### Integration and regression tests

- nominal grasp survives a two-second zero-action hold for 100 of 100 resets;
- held-out perturbed validation achieves at least 95% stability;
- the first policy step has no joint-target discontinuity;
- a deliberately unstable grasp is rejected before the policy sees it;
- existing `orca_sim` and `orca_train` test suites continue to pass.

The 100-reset and robust held-out evaluations are acceptance commands rather
than part of every fast unit-test invocation.

## Spark rollout sequence

1. Implement and pass local unit tests and short deterministic physics checks.
2. Sync isolated code revisions to Spark.
3. Run CEM search and held-out grasp validation on Spark.
4. Pull the selected grasp artifact and report back locally for inspection.
5. Start curriculum stage 0 on Spark only after all grasp acceptance gates pass.
6. Promote through small-angle stages using the fixed evaluation criteria.
7. Start 90-degree training only after stage 3 meets both success and drop-rate
   requirements.

No long local training or full local search is part of this workflow.

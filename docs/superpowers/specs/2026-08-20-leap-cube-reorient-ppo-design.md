# ORCA LeapCubeReorient PPO Design

**Date:** 2026-08-20  
**Status:** Approved design; pending document review  
**Repositories:** `orca_sim`, `orca_train`  
**Upstream reference:** MuJoCo Playground `v0.2.0`

## Problem statement

The existing ORCA cube-orientation training path uses a CPU Gymnasium
environment and an off-policy deterministic actor-critic implementation. It
does not reproduce the tuned `LeapCubeReorient` PPO setup, and the current v2
task begins with the cube on an open palm rather than in a stable grasp. A
previous rollout showed that the cube drops after about 0.48 seconds and that a
policy can temporarily reduce orientation error while the cube is falling.

The requested result is an ORCA policy trained with the official
`LeapCubeReorient` Brax PPO configuration and asymmetric actor-critic
observations. The only task-level departures are those required by the ORCA
embodiment: a fixed wrist, a fifth fingertip in the privileged observation, a
validated stable-grasp reset, and a stability-first orientation curriculum.

The upstream references are:

- [PPO configuration](https://github.com/google-deepmind/mujoco_playground/blob/v0.2.0/mujoco_playground/config/manipulation_params.py)
- [LeapCubeReorient environment](https://github.com/google-deepmind/mujoco_playground/blob/v0.2.0/mujoco_playground/_src/manipulation/leap_hand/reorient.py)
- [Brax PPO runner](https://github.com/google-deepmind/mujoco_playground/blob/v0.2.0/learning/train_jax_ppo.py)

## Relationship to the grasp-stability design

This design retains the following parts of
`2026-08-19-grasp-stability-design.md`:

- the named palm-center definition;
- the transferable place-close-hold preparation sequence;
- robust CEM grasp-pose search;
- held-out grasp validation and the 95% acceptance gate;
- hash-verified grasp artifacts and structured reset diagnostics.

It supersedes the previous design for the rotation learner, policy
observations, reward, action semantics, curriculum terminal stage, checkpoint
format, and training loop. In particular, the new path uses on-policy Brax PPO
and the Leap reward rather than the existing TD3-style state agent and custom
grasp-aware reward.

## Goals

1. Reuse the official `LeapCubeReorient` Brax PPO implementation and tuned
   configuration without locally redefining PPO behavior.
2. Preserve the official 57-dimensional policy observation exactly.
3. Preserve the semantic privileged observation while adding ORCA's fifth
   fingertip, producing 131 dimensions.
4. Fix `right_wrist` and train the 16 finger actuators.
5. Start all PPO rollouts from a validated, physically attainable grasp.
6. Advance from holding through full SO(3) reorientation within one
   checkpointed 200-million-step run.
7. Train and evaluate the final model on Spark using GPU-batched MuJoCo
   Warp/MJX.

## Non-goals

- Autonomous pickup is not part of the task.
- The policy does not receive contact force, motor current, or other new
  deploy-only inputs.
- No invisible support, weld, gravity change, or other retention aid is used.
- No PPO tuning is performed silently. Any deviation from the upstream
  configuration must be an explicit diagnostic run and cannot be called the
  final model.
- Torch, ONNX, or hardware-controller deployment is not required for the
  training acceptance gate. The delivered policy bundle must contain enough
  information for a later deployment adapter.

## Integration choice

Three integration approaches were considered:

1. Fork MuJoCo Playground and add ORCA to its global registry.
2. Reimplement PPO and the Playground environment API inside `orca_train`.
3. Add a thin ORCA environment and runner adapter that imports the pinned
   Playground implementation.

The third approach is selected. A fork would create a long-lived merge burden,
while a local PPO rewrite would make exact parity difficult to demonstrate.
The thin adapter keeps ORCA code in ORCA repositories and uses upstream Brax
PPO directly.

`playground` is an optional dependency group so the existing Torch training
commands remain usable. The dependency is pinned to the upstream `v0.2.0` tag
and the resolved commit and lockfile hash are recorded in every run manifest.

## Repository boundaries

### `orca_sim`

- Add a named `right_palm_grasp_center` site.
- Add named thumb, index, middle, ring, and pinky fingertip sites.
- Retain the CPU place-close-hold reset and CEM grasp search as the source of
  the validated grasp artifact.
- Add or expose an MJX-compatible cube scene with a goal mocap body and all
  required named model elements.
- Keep the generic CPU task available independently of Playground.

### `orca_train`

- Add the ORCA Playground environment and model-name constants.
- Add the exact PPO configuration adapter.
- Add the six-stage curriculum and checkpointed curriculum state.
- Add Playground training, evaluation, checkpoint, export, and rollout-video
  entry points.
- Validate the grasp artifact and observation schema before compilation or
  training.

## Embodiment mapping

The ORCA v2 model has 17 position actuators: one wrist actuator and 16 finger
actuators. The environment exposes a 16-dimensional action to PPO and builds a
17-dimensional MuJoCo control vector internally. The wrist target is always
the validated grasp-artifact wrist position and is never modified by policy
actions.

The active actuator order is the existing ORCA model order with
`right_wrist_actuator` removed. The order, joint names, control ranges, grasp
artifact hash, and observation schema version are stored in checkpoints and
exports. Loading fails if any of them differ.

The five fingertip positions are measured from named sites attached to the
thumb, index, middle, ring, and pinky distal bodies. They are expressed
relative to `right_palm_grasp_center`, matching the Leap privileged-observation
semantics.

## Environment timing and actions

The environment configuration copies `LeapCubeReorient`:

- control timestep: 0.05 seconds;
- simulation timestep: 0.01 seconds;
- action repeat: 1;
- action scale: 0.5;
- EMA alpha: 1.0;
- episode length: 1000 policy steps;
- success threshold: 0.1 radians;
- observation history length: 1;
- default implementation: `warp`;
- `naconmax`: `30 * 8192`;
- `njmax`: 160.

For active finger actuator `i`:

```text
target[i] = clip(previous_ctrl[i] + 0.5 * action[i], ctrl_low[i], ctrl_high[i])
```

The wrist target is inserted unchanged. No extra residual-action envelope is
applied in the final parity run.

## Stable-grasp reset

Grasp search runs before PPO. It fixes the wrist and optimizes the 16 finger
joints using the transferable place-close-hold sequence. The chosen artifact
must pass:

- 100 of 100 nominal two-second hold resets;
- at least 95% held-out perturbed resets;
- palm plus at least two distinct finger contacts;
- palm-relative drift and velocity limits;
- actuator effort and contact-force limits;
- schema and content-hash validation.

The expensive closing trajectory is not replayed inside every batched MJX
reset. Instead, the post-hold physical state from the validated artifact is the
ORCA equivalent of Leap's `home` keyframe. A training reset then performs the
Leap operations around that state:

1. hold the wrist at the artifact value;
2. sample each finger joint from `grasp_qpos + 0.1 * Normal(0, 1)` and clip it
   to its control range;
3. sample cube position within `Uniform(-0.01, 0.01)` metres on each axis
   around the artifact cube center;
4. sample a uniform cube quaternion;
5. set all initial joint, cube linear, and cube angular velocities to zero;
6. initialize the finger controls to the sampled finger joint positions.

The exact training reset distribution is included in held-out grasp
validation. PPO does not start if it fails the 95% gate. The physical takeover
workflow still runs the place-close-hold sequence and initializes the policy
control target from the measured post-hold joint state.

## Asymmetric observations

### Actor: `state`, 57 dimensions

The order exactly follows Leap:

| Slice | Field | Dimensions |
|---|---|---:|
| 0 | noisy active finger joint angles | 16 |
| 1 | noisy joint angle minus motor target, history length 1 | 16 |
| 2 | palm position minus noisy cube position, history length 1 | 3 |
| 3 | noisy cube-to-goal relative rotation matrix `ravel()[3:]` | 6 |
| 4 | previous policy action | 16 |
| | Total | 57 |

Actor noise also follows Leap:

- joint position: uniform noise scaled by 0.05;
- cube position: uniform noise scaled by 0.02 metres;
- cube quaternion: additive Gaussian noise scaled by 0.1, then normalized;
- random pose injection probability: 0.0.

### Critic: `privileged_state`, 131 dimensions

The critic begins with the complete 57-dimensional actor state, including its
noise, then appends:

| Field | Dimensions |
|---|---:|
| clean active finger joint positions | 16 |
| clean active finger joint velocities | 16 |
| five fingertip positions relative to the palm | 15 |
| clean palm-to-cube position error | 3 |
| clean cube-to-goal rotation error | 6 |
| cube linear velocity | 3 |
| cube angular velocity | 3 |
| perturbation direction | 6 |
| applied cube wrench | 6 |
| Actor plus appended fields | 131 |

The fifth fingertip is the only dimensional difference from Leap's
128-dimensional four-finger privileged state.

## Reward and termination

The reward terms and scales copy Leap:

| Term | Scale |
|---|---:|
| orientation tolerance | 5.0 |
| palm-cube position tolerance | 0.5 |
| termination | -100.0 |
| squared deviation from grasp home pose | -0.5 |
| first- and second-order action rate | -0.001 |
| normalized joint velocity | 0.0 |
| absolute velocity-times-actuator-force energy | -0.001 |
| success bonus | 100.0 |

The weighted dense reward is multiplied by the 0.05-second control timestep.
The orientation tolerance uses target interval `[0, 0.2]`, margin `pi`, and a
linear sigmoid. The position tolerance uses target interval `[0, 0.02]`,
margin `0.05`, and a linear sigmoid.

Termination occurs on non-finite state or after the cube falls 0.10 metres
below its nominal grasp-center height. This is the palm-frame equivalent of
Leap's start height `0.05` and termination height `-0.05`; it avoids coupling
the threshold to ORCA's world mount offset.

No additional grasp-quality gate or custom falling-progress reward is added.
The stable reset, termination, and curriculum are the approved task-level
adaptations.

## Domain randomization

The source behavior and intervals are preserved and applied to corresponding
ORCA elements:

- all five fingertip sliding frictions: `Uniform(0.5, 1.0)`;
- cube inertia scale: `Uniform(0.8, 1.2)`;
- cube inertial-position offset: `Uniform(-5 mm, 5 mm)` on each axis;
- active finger `qpos0` offset: `Uniform(-0.05, 0.05)` radians;
- active finger friction loss scale: `Uniform(0.5, 2.0)`;
- active finger armature scale: `Uniform(1.0, 1.05)`;
- hand-link mass scale: `Uniform(0.9, 1.1)`;
- active actuator stiffness scale: `Uniform(0.8, 1.2)`;
- active joint damping scale: `Uniform(0.8, 1.2)`.

The upstream implementation's actual arrays, not comments about intended
randomization, define parity. The fixed wrist is excluded from policy-side
joint and actuator randomization.

External cube perturbations remain disabled by default, matching Leap. Their
direction and applied wrench remain present in the privileged state as zeros.

## Curriculum

Curriculum state is part of the environment state and every checkpoint. It is
not implemented by restarting PPO or changing PPO hyperparameters.

| Stage | Goal distribution | Promotion criterion |
|---|---|---|
| 0: hold | current cube orientation for 2 seconds | success >= 99%, drop <= 1% |
| 1: 10-15 deg | random axis, angle uniformly 10-15 degrees | success >= 80%, drop <= 5% |
| 2: 30 deg | random axis, 30 degrees | success >= 80%, drop <= 5% |
| 3: 60 deg | random axis, 60 degrees | success >= 80%, drop <= 5% |
| 4: 90 deg | random axis, 90 degrees | success >= 80%, drop <= 5% |
| 5: full SO(3) | Leap's uniform reset goal and success goal update | terminal stage |

Promotion requires two consecutive passing evaluations on fixed held-out seed
sets. The final acceptance report applies the 80% success and 5% drop limits to
stage 5 even though it has no further promotion.

The complete run remains capped at 200 million environment steps. If it does
not reach and pass full SO(3) evaluation within that budget, it is reported as
an unsuccessful training attempt rather than relabelled as a completed model.

## PPO parity

The final run uses the upstream `brax_ppo_config("LeapCubeReorient")` values:

- 200,000,000 timesteps;
- 20 evaluations;
- 8192 environments;
- batch size 256;
- unroll length 40;
- 32 minibatches;
- 4 updates per batch;
- discount 0.99;
- learning rate `3e-4`;
- entropy cost `1e-2`;
- observation normalization enabled;
- reward scaling 1.0;
- one reset per evaluation;
- policy and value MLPs `(512, 256, 128)`;
- policy key `state` and value key `privileged_state`.

All unspecified PPO fields come from the same pinned Brax/Playground runner.
The adapter must not introduce local fallback defaults. A parity test compares
the serialized effective ORCA PPO config against the serialized upstream Leap
config field by field.

Reduced environment counts or timesteps are allowed only for compile and
smoke tests. Their manifests are marked `diagnostic` and their checkpoints
cannot be selected as the final model.

## Training pipeline

1. Run local unit tests and deterministic CPU physics checks.
2. Build and validate the stable-grasp artifact on Spark.
3. Install the pinned Playground/JAX/MuJoCo Warp environment in an isolated
   remote project environment.
4. Verify that JAX reports a GPU backend on the NVIDIA GB10.
5. Compile the ORCA model for Warp/MJX and run batched reset and step tests.
6. Run a reduced PPO smoke test and verify finite losses, rewards, gradients,
   observations, and checkpoints.
7. Launch the exact 8192-environment, 200-million-step run in a named persistent
   remote session.
8. Monitor metrics and stage transitions; preserve failure logs and best
   checkpoints.
9. Evaluate the best full-SO(3) checkpoint on held-out seeds.
10. Export and copy the accepted artifacts back to the local workspace.

No full training run is launched before model compilation, reset validation,
and PPO smoke tests pass.

## Checkpoints and artifacts

Each checkpoint or export records:

- ORCA repository revisions;
- pinned Playground version and resolved revision;
- effective environment and PPO configs;
- observation and action schema versions;
- active joint and actuator order;
- fixed wrist position;
- grasp artifact and validation-report hashes;
- curriculum stage, promotion streak, and held-out seed-set version;
- policy parameters, value parameters, optimizer state, and observation
  normalizer state;
- training step, random seed, and diagnostic/final classification.

The final delivery contains:

- the best restorable training checkpoint;
- a policy-only inference bundle including normalization statistics;
- grasp artifact and validation reports;
- training metrics and curriculum transition report;
- final held-out evaluation report;
- a rendered representative rollout video;
- exact commands and run manifest needed to reproduce evaluation.

## Error handling

- Missing, invalid, or hash-mismatched grasp artifacts stop training.
- A model-name, actuator-order, observation-shape, or checkpoint-schema
  mismatch fails before JIT compilation.
- Reset validation has bounded attempts and preserves failure reasons.
- NaN or non-finite state, observation, reward, loss, gradient, or parameter
  terminates the affected diagnostic or training run and writes a failure
  manifest.
- Failure to obtain a JAX GPU backend or compile the model on GB10 blocks full
  training and is reported with version and driver diagnostics.
- Failure to meet the grasp gate blocks PPO.
- Failure to meet the final full-SO(3) gate blocks the `accepted` label.

## Test strategy

Implementation follows test-driven development.

### Fast tests

- exact PPO config parity with upstream Leap;
- fixed wrist and 16-dimensional action mapping;
- exact 57-dimensional actor field order and noise behavior;
- exact 131-dimensional privileged field order;
- reward and termination formulas on constructed states;
- relative action update and clipping;
- domain-randomization field selection and bounds;
- deterministic curriculum promotion and checkpoint round trip;
- grasp artifact hash and schema mismatch rejection;
- diagnostic checkpoints cannot be accepted as final artifacts.

### Physics and integration tests

- MuJoCo CPU and Warp/MJX model loading;
- scalar and batched reset/step shape checks;
- finite observations, rewards, and gradients;
- wrist target invariance over rollouts;
- zero-action grasp hold and perturbed reset acceptance;
- one shortened PPO update and checkpoint restore;
- existing `orca_sim` and `orca_train` tests remain green.

### Final acceptance gates

- nominal grasp: 100/100 two-second holds;
- held-out perturbed grasp stability: at least 95%;
- full run: exactly 200 million environment steps with the parity PPO config;
- final stage: full SO(3);
- held-out success: at least 80%;
- held-out drop rate: at most 5%;
- restorable checkpoint and policy-only export produce matching deterministic
  actions on a fixed observation batch;
- delivered video and report use the accepted checkpoint.


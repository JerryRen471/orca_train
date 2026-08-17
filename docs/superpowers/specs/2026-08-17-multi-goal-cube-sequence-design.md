# Multi-Goal Cube Sequence Design

## Objective

Train one goal-conditioned DrQ-v2 policy to complete a continuous sequence of in-hand cube reorientation tasks. An episode ends only when the hand completes 20 targets, drops the cube, or exceeds the time limit for one target. This prevents a transient red-face-up pose from ending the episode before the subsequent cube motion reveals a drop.

## Target Sequence

The target is the desired world-space normal of the cube's red face. The target set contains the six unit axis directions: positive and negative X, Y, and Z.

Targets use a shuffled bag. Each bag contains all six directions in random order. After consuming a bag, the environment shuffles a new one and swaps its first entry when necessary so it cannot equal the preceding target. The first target also excludes the direction already satisfied by the initial cube pose. This produces a random but balanced sequence without consecutive duplicates or free initial completions.

## Observation

The visual observation remains three stacked 84 by 84 RGB frames. The proprioceptive vector grows from 17 to 20 values:

- 17 joint angles normalized by their configured joint ranges.
- 3 components of the current target direction in world coordinates.

The agent and replay buffer already derive their observation dimensions from the environment, so they must accept the new 20-value vector without a fixed-size special case. Existing checkpoints are incompatible with the changed input dimension; multi-goal training starts from scratch.

## Task State and Step Ordering

The ORCA Sim environment owns the target sequence and maintains:

- Current target direction.
- Number of completed targets.
- Steps elapsed for the current target.
- Consecutive stable-success steps.
- Cached completion, drop, timeout, and episode-success events for the current step.

Each environment step performs the following operations exactly once:

1. Apply the action and advance MuJoCo.
2. Increment the current target timer.
3. Test drop first.
4. If not dropped, test the current target against the stable-success conditions.
5. Complete the target only after ten consecutive qualifying steps.
6. If fewer than 20 targets have been completed, draw the next target and reset the per-target timer and stability counter.
7. Build the returned observation using the next target, while keeping the step reward associated with the target just completed.

Stable success requires all existing constraints:

- Target direction error at most 15 degrees.
- Cube height at least 0.12 metres.
- Cube linear speed at most 0.15 metres per second.
- Cube angular speed at most 2 radians per second.
- All conditions held for ten consecutive environment steps.

Repeated reward, termination, observation, and info queries only read cached events and never advance counters.

## Reward and Termination

For the active target, the progress reward is:

`5 * (current_alignment - previous_alignment) - 0.01`

Completing any target adds 10. After a target switch, the environment initializes the previous-alignment baseline against the new target at the current physical state. Therefore, changing the target cannot create a false progress reward.

The episode ends under exactly one of these conditions:

- `sequence_complete`: the twentieth target is completed; the final target bonus is retained.
- `dropped`: cube height falls below 0.10 metres; subtract 10.
- `task_timeout`: the active target reaches 200 steps without completion; no additional timeout penalty is applied.

Drop has priority when multiple conditions occur in the same step. Target completion before target 20 does not terminate or truncate the episode. A task timeout is a terminal task failure rather than a global time-limit truncation. The maximum possible episode length is 4,000 environment steps.

## Configuration and Diagnostics

Training configuration exposes the sequence length and per-target time limit, defaulting to 20 and 200. Existing stability, height, velocity, reward, and drop thresholds remain configurable and are recorded in `config.json`.

Environment `info` includes:

- `target_direction`
- `target_completed`
- `completed_target_direction` and `completed_target_steps` when a target completes
- `tasks_completed`
- `task_elapsed_steps`
- `termination_reason`, one of `sequence_complete`, `dropped`, `task_timeout`, or `null`
- Existing alignment, cube pose, velocity, stability, success, and drop diagnostics

`is_success` means the complete 20-target sequence succeeded. A per-target completion is reported separately through `target_completed`; its completed direction and duration remain available even when the returned observation has already switched to the next target.

## Training and Evaluation

DrQ-v2, action scaling, joint-range normalization, replay, and checkpoint cadence remain unchanged. Training begins from a new replay buffer and randomly initialized networks.

Evaluation reports:

- Mean targets completed per episode.
- Full-sequence success rate.
- Total targets completed across evaluation episodes.
- Drop rate.
- Task-timeout rate.
- Mean steps per completed target.
- Mean episode return.

Training episode records also include completed-target count and termination reason. The current single-target Spark run remains untouched until the multi-goal implementation passes verification and a separate training launch is approved.

## Verification

Automated tests cover:

- Every shuffled bag contains all six directions exactly once.
- Bag boundaries and initial targets never repeat an already satisfied direction.
- The target direction is appended to the normalized joint vector.
- Ten consecutive stable steps complete exactly one target.
- An unstable step resets the hold counter.
- Targets 1 through 19 switch goals without ending the episode.
- Target 20 ends with `sequence_complete`.
- Target switching cannot generate artificial progress reward.
- The active target times out exactly at 200 steps.
- A drop overrides simultaneous completion or timeout.
- Evaluation aggregates sequence success, target counts, drop, timeout, and target duration correctly.

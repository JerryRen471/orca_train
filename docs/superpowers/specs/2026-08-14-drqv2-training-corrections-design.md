# DrQ-v2 Training Corrections Design

## Goal

Make cube-flip training optimize actual successful flips, remove hidden control state, and restore the important DrQ-v2 training mechanics that are missing from the current pipeline.

## Environment and reward

- Use reward `5 * (alignment_now - alignment_previous) + 10 * success - 10 * dropped - 0.01`.
- Terminate on success or drop and truncate after 200 steps.
- Map each normalized action to at most 3 degrees relative to the current measured joint angles, then clip to the joint ROM. Do not accumulate an unobserved actuator target.
- Normalize the 17 measured joint angles independently by their ROM into `[-1, 1]`.

## Agent and replay

- Use three-step discounted returns, flushing shorter returns at episode boundaries.
- Increase replay capacity to 200,000 transitions.
- Sample differentiable noisy actor actions and optimize against `min(Q1, Q2)`.
- Preserve three stacked 84×84 RGB frames, 17-dimensional proprioception, 17-dimensional free-wrist actions, random-shift augmentation, target critics, and the existing exploration schedule.

## Evaluation and observability

- Evaluate every checkpoint on the same fixed set of 20 seeds.
- Log success rate, drop rate, return, Q estimates, target Q, and critic loss.
- Train from scratch for 1,000,000 steps on Spark after local tests and a short CUDA smoke run pass.

## Resource bounds and compatibility

- Keep replay in RAM; 200,000 transitions fit within the Spark host's 121 GiB memory.
- Preserve checkpoint loading for newly produced checkpoints; older checkpoints remain usable only with their matching historical environment wrapper.
- Do not overwrite previous run directories.


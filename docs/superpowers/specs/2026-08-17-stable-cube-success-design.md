# Stable Cube Success Design

## Goal

Prevent a falling, fast-spinning cube from being counted as a successful in-hand orientation.

## Success state

After each MuJoCo environment step, compute one success candidate from the resulting state:

- Red-face error is at most 15 degrees.
- Cube center height is at least 0.12 m.
- Cube linear speed is at most 0.15 m/s.
- Cube angular speed is at most 2.0 rad/s.

Increment a consecutive-stability counter once when all conditions hold; otherwise reset it to zero. Success requires 10 consecutive candidate steps. The counter is updated only once inside `step()`, never inside reward, termination, or info accessors.

## Failure and reward

- Raise the drop threshold from 0.05 m to 0.10 m.
- Drop takes priority over success and resets the stability counter.
- Award the success bonus and terminate only after the counter reaches 10.
- Preserve progress reward, step penalty, drop penalty, and all corrected DrQ-v2 mechanics.

## Verification and training

- Test transient orientation, counter reset, exact tenth-step success, drop priority, and single-update behavior.
- Run both repository test suites and a Spark CUDA smoke test.
- Start a new one-million-step run from scratch in a new output directory.


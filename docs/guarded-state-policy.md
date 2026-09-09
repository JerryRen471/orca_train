# Guarded state policy

`GuardedStatePolicy` combines a trained `StateAgent` with joint target projection
and a goal hold controller. It uses the existing state observation, including
the relative target quaternion and normalized controller targets. It does not
require a simulator reference or contact information at inference time.

The default controller leaves 5% of each joint's ROM at both ends. When target
orientation error is at most 10°, it stops accumulating target increments. The
task still decides success using its own tolerance, speed, contact and hold-time
requirements. Choose a controller hold threshold inside the task's tolerance.
For the fixed 90° experiment, the task requires error at most 15° and a stable
2-second hold within 8 seconds.

```python
from orca_train.guarded_state_policy import GuardedStatePolicy

policy = GuardedStatePolicy(
    agent,
    joint_low=joint_low,
    joint_high=joint_high,
    active_actuator_indices=active_indices,
    max_delta_radians=max_delta_radians,
    joint_target_margin_fraction=0.05,
    hold_error_degrees=10.0,
)
action = policy.act(observation, step=0, eval_mode=True)
```

The joint ranges and actuator order must match the environment that encoded the
state. The observation layout is `3 * number_of_actuated_joints + 13`; active
actions may exclude fixed joints such as the wrist. Store these controller
parameters with the checkpoint and apply the wrapper during deployment as well
as evaluation. `train_state` does not automatically enable the wrapper.

An initially outside target moves toward the interior at the original bounded
per-step increment. The guard limits commanded targets; dynamics can still take
measured joint positions closer to a limit. The reported measured-limit fraction
must therefore be checked in rollouts.

`value()` queries the smaller critic estimate for the guarded action actually
executed. The critics were trained with the source policy and may be miscalibrated
after adding the controller. Compare estimates with realized discounted returns;
successful task completion does not establish accurate value estimates.

# Pixel student domain randomization

`orca_train.pixel_distill collect/evaluate --domain-config configs/domain_randomization/transfer_v1.json`
enables episode-level simulation and sensor randomization. Omitting the flag preserves nominal behavior.
Students still consume only three 84 x 84 RGB frames; the teacher sees state only when generating action labels.

The existing scene camera automatically points at the cube (`targetbody` mode).
For a fixed physical camera, pass `--camera-look-at 0.177 -0.015 0.175` to collection
and evaluation. This sets the orientation once using a constant world-space
reference, disables target tracking, and leaves orientation fixed when the cube
moves. The reference is a provisional mount setup, not a real-camera calibration.
Position randomization translates this fixed camera without retargeting it.
The selected camera setup is recorded separately from the domain profile in data
manifests and evaluation reports; neither setup is provided to the student as input.

## Provisional ranges

| Quantity | Distribution |
| --- | --- |
| Cube mass and inertia | Same uniform multiplier 0.90 to 1.10 (72 to 88 g for the current cube) |
| Contact friction | Common uniform multiplier 0.90 to 1.10 for all geom friction components |
| Position actuator stiffness | Independent uniform multiplier 0.95 to 1.05 per actuator; target and position terms scaled together |
| Joint viscous damping | Independent uniform multiplier 0.90 to 1.10 per DOF; nominal zero remains zero |
| Camera position | Independent uniform displacement -8 to +8 mm per Cartesian axis |
| Camera FOV | Uniform 28 to 32 degrees around the student's 30-degree setting |
| Light diffuse intensity | Uniform multiplier 0.75 to 1.25 per light |
| Light ambient intensity | Uniform addition 0 to 0.025 per light |
| Material RGB | Independent uniform multiplier 0.85 to 1.15 per channel, clipped to [0, 1] |
| Unmaterialized geometry RGB | Uniform multiplier 0.85 to 1.15 per geometry; face identity retained |
| Exposure | Uniform multiplier 0.85 to 1.15 |
| White balance | Independent uniform multiplier 0.92 to 1.08 per channel |
| Sensor noise | Episode standard deviation uniform 0 to 1.5 uint8 levels; fresh independent Gaussian noise per captured frame |
| Blur | Episode mixture uniform 0 to 0.20 of a 3 x 3 box-filtered image |

All distributions are sampled independently unless sharing is stated. Physical and visual parameters stay fixed throughout grasp preparation and the episode. Each frame is corrupted once before stacking, so repeated history frames remain identical. Material changes preserve alpha and do not accidentally override assigned materials via geometry colors. Background texture layout is unchanged, but its material color and illumination vary.

These ranges are provisional engineering assumptions, not measured hardware tolerances. Camera rotation, lens distortion, occlusion, communication delay, backlash, motor torque saturation and hand geometry are not randomized by this profile. The cube face marker and the prepared starting grasp are still required. Do not infer hardware readiness from randomized simulation scores.

## Reproduction

Run Python through `uv` in the configured project environment. `TEACHER` is an exported state teacher directory, `BASELINE` a previously trained image-only checkpoint, and `NOMINAL_DATA` its teacher demonstration directory.

```bash
uv run python -m orca_train.pixel_distill collect \
  --teacher "$TEACHER" --episodes 96 --seed 300000 \
  --domain-config configs/domain_randomization/transfer_v1.json --output runs/dr_data
uv run python -m orca_train.pixel_distill fit \
  --data "$NOMINAL_DATA" runs/dr_data --resume "$BASELINE" \
  --seed 1 --steps 3000 --learning-rate 0.0001 --device mps --output runs/dr_student
uv run python -m orca_train.pixel_distill evaluate \
  --teacher "$TEACHER" --student runs/dr_student/checkpoint_3000.pt \
  --episodes 50 --seed 330000 --domain-config configs/domain_randomization/transfer_v1.json \
  --output runs/dr_evaluation.json
```

Use distinct seed ranges for each angle and for training, development, validation and independent testing. Evaluate the frozen model in nominal conditions as well by omitting `--domain-config`. Compare the previous student on the same held-out seeds and domain profile. Check teacher success under the same physical draws before relying on its labels. Collection retains failed episodes and records their outcomes; it never silently resamples a difficult domain.

Every reset first restores a snapshot of the nominal model, applies its seed-derived perturbations, refreshes MuJoCo constants, and then runs the original grasp preparation. Separate random streams for physics, appearance and sensor noise leave the task RNG and target selection untouched. The task's timing, success conditions, rewards, joint ranges and accumulated target action interface are unchanged. No episode parameter is added to student observations.

Data manifests and evaluation reports contain the full configuration and sampled parameters per episode. Checkpoints bind hashed manifests and source files; fine-tuning also retains the parent's training seeds to prevent evaluation leakage. Dataset sampling is uniform over episodes: mixing 64 nominal and 96 randomized demonstrations gives 40% nominal and 60% randomized samples in expectation. Fresh Adam optimizes teacher-action MSE; this is supervised fine-tuning, not reinforcement learning or real-robot training.

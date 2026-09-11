# Image-only teacher distillation

`PixelStudent` consumes only `pixels`: three consecutive 84x84 RGB frames in
CHW order, stacked to `(9, 84, 84)`, with dtype `uint8`. It reuses the visual
encoder and actor building blocks from DrQ-v2, with supervised teacher-action
regression. Each checkpoint is a separate fixed-target policy. No joint angles,
cube pose, numeric goal, timestep, action history, or critic are inference inputs.

`PixelView` renders an existing `OrcaStateCubeEnv` without changing its physics,
reset preparation, or accumulated-target delta action semantics. The prior
`OrcaVisualCubeEnv` includes proprioception and uses different delta semantics;
it is not the environment adapter for these students.

During collection only, the teacher can inspect the privileged observation.
Labels correspond to the image immediately before the action, including when
a student executes a different action during DAgger. The image stack never
includes future frames and is reset by repeating the first image of the episode.
The 90-degree teacher's joint projection and goal holding are included in its
action labels; the student does not call that state-based wrapper at inference.
The original low-level hard joint target limits remain part of the actuator
interface, as they are for the teacher.

Use the `uv` environment and matching orca_train/orca_sim source checkouts:

```sh
uv run --no-sync python -m orca_train.pixel_distill collect \
  --teacher /path/to/trained_turn90_policy --episodes 64 --seed 201000 \
  --output /path/to/data90
uv run --no-sync python -m orca_train.pixel_distill fit \
  --data /path/to/data90 --seed 1 --steps 1500 --device mps \
  --output /path/to/student90_seed1
uv run --no-sync python -m orca_train.pixel_distill evaluate \
  --teacher /path/to/trained_turn90_policy \
  --student /path/to/student90_seed1/checkpoint_1500.pt \
  --episodes 100 --seed 211000 --output /path/to/validation90.json
```

The evaluation `--teacher` directory supplies the task configuration. Teacher
weights are loaded only with `--teacher-policy`; ordinary student evaluation
does not load or call the teacher. For a standalone student deployment, retain
the exact task and camera configuration alongside the inference checkpoint.

If closed-loop cloning drifts, collect teacher labels on student-visited states
with `collect --student ... --beta 0.2`; beta is the per-step probability of
executing the teacher action. Then `fit --resume ... --data DATA DAGGER_DATA ...`
continues the actor with a fresh optimizer. Keep reset seeds disjoint between
data collection, validation, and independent tests. Datasets retain hashes,
teacher identity, camera configuration, student parent, action noise, and mixing
probability. Mixing different teachers or duplicate reset seeds is rejected.

Select checkpoints by complete task rollouts on validation seeds and freeze
their hashes before independent evaluation. Report all initialization seeds,
not just the best model. Action loss is not a substitute for task success.
The checkpoint contains only encoder/actor parameters and metadata, so inference
requires neither teacher nor critic weights:

```python
from orca_train.pixel_student import PixelStudent

student, metadata = PixelStudent.load("student.pt", device="cpu")
action = student.act({"pixels": stacked_rgb_frames})
```

# Fixed-camera domain randomization experiment

The scene's `closeup` camera originally used MuJoCo `targetbody` mode, so it turned
toward the cube. A fixed-camera check exposed a transfer gap, especially for the
90-degree visual student. The final students were therefore given actual
fixed-camera demonstrations with physical and visual randomization.

All results below use a camera aimed once at the constant world point
`(0.177, -0.015, 0.175)`. It never follows the cube. The random profile translates
the camera between episodes, keeping its orientation fixed.

## Independent results

Each model ran 50 nominal and 50 randomized episodes with a fixed camera.

| Target | Training seed | Nominal success | Randomized success | Randomized drops |
| --- | --- | --- | --- | --- |
| 60 degrees | 1 | 43/50 | 44/50 | 0/50 |
| 60 degrees | 2, recommended before testing | 50/50 | 47/50 | 0/50 |
| 60 degrees | 3 | 50/50 | 46/50 | 0/50 |
| 90 degrees | 1, recommended before testing | 49/50 | 41/50 | 0/50 |
| 90 degrees | 2 | 49/50 | 41/50 | 0/50 |
| 90 degrees | 3 | 49/50 | 41/50 | 1/50 |

The original seed-1 students scored 25/50 and 1/50 on the same fixed-camera
randomized scenarios, with zero and six drops respectively. Comparing the same
seed-1 model ancestry, final success improved to 44/50 and 41/50. The 90-degree
target/actual limit sampling fractions decreased from 7.32%/4.47% to 0.30%/0.19%.

Success still requires completing the turn within an 8-second budget, 15-degree
tolerance, reliable grasp and velocity limits, plus 2 seconds of stable hold.
The wrist is fixed and the task uses the prepared `thumb_opposed_v1` grasp and
positive hand-X rotation. Limit proximity means the outer 1% of the hard joint
range, averaged over active joints and control steps.

## Training and separation

Each of six existing RGB-only students received 3000 supervised updates on 64
original nominal plus 96 randomized teacher episodes. A second stage added 96
fixed-camera randomized episodes per angle, mixed them with the original 64
nominal episodes, and continued each corresponding model for 2000 updates.
The second-stage mixture still includes the original tracking-camera examples.
Both stages use teacher-action MSE, fresh Adam at 1e-4, batch size 64 and random
shift padding 2. This is 30000 additional updates across the six models, without
student critics or reinforcement learning.

Final fixed-camera collection uses seeds 400000-400095 / 401000-401095. Fixed
camera development uses 410000-410015 / 411000-411015, validation uses
420000-420015 / 421000-421015, and independent testing uses
430000-430049 / 431000-431049. Earlier-stage independent seeds are separate.
All failed demonstrations are retained, and ancestor training seeds remain
excluded from evaluation after fine-tuning.

All six final checkpoints and the recommended seed per angle were fixed before
independent testing. Recommendations prefer nominal validation success >=95%,
then lower randomized drop rate, higher randomized success, higher nominal
success, and finally lower seed. All models are reported; seed 1 is weaker for
nominal 60-degree control and is not the recommended candidate.

Models share the same evaluation seeds within each angle, so repeated model
evaluations are correlated and are not additional independent scene samples.
The independent set was not used to retune or reselect models.

## Scope

The [profile and commands](../pixel-domain-randomization.md) document the
provisional mass, inertia, friction, gain, damping, camera, light, material,
exposure, white balance, noise and blur ranges. Students receive only three
84x84 RGB frames spaced 80 ms apart. The camera setup is not a policy input.

130 tests pass, including fixed-camera invariance to cube movement, parameter
restoration, independent random streams, actuator equilibrium, mass/inertia
consistency, material precedence, unchanged physics under visual-only changes,
RGB-only inference, label timing and inherited seed exclusion.

The camera pose and physical ranges have not been measured on hardware. Camera
rotation perturbations, distortion, occlusion, latency, backlash, actuator
saturation and geometry variation remain unvalidated. The state teachers also
fail some physical randomizations. These are simulation results and do not
establish real-robot readiness.

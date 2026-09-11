from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


def configure_fixed_camera(model, camera_name, look_at):
    """Set a world-mounted camera once, without following the simulated object."""
    import mujoco

    camera = model.camera(camera_name).id
    target = np.asarray(look_at, dtype=np.float64)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("Camera look-at must contain three finite world coordinates")
    if model.cam_bodyid[camera] != 0:
        raise ValueError("Fixed camera configuration requires a world-mounted camera")
    z = model.cam_pos[camera] - target
    if np.linalg.norm(z) < 1e-8:
        raise ValueError("Camera position and look-at must differ")
    z /= np.linalg.norm(z)
    up = np.array([0., 1., 0.]) if abs(z[2]) > .99 else np.array([0., 0., 1.])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mujoco.mju_mat2Quat(model.cam_quat[camera], np.column_stack([x, y, z]).ravel())
    model.cam_mode[camera] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    model.cam_targetbodyid[camera] = -1


@dataclass(frozen=True)
class DomainConfig:
    """Provisional symmetric ranges around the calibrated simulation, per episode."""

    mass_fraction: float = 0.10
    friction_fraction: float = 0.10
    gain_fraction: float = 0.05
    damping_fraction: float = 0.10
    camera_position_m: float = 0.008
    camera_fovy_deg: float = 2.0
    light_fraction: float = 0.25
    material_fraction: float = 0.15
    exposure_fraction: float = 0.15
    white_balance_fraction: float = 0.08
    noise_std_max: float = 1.5
    blur_mix_max: float = 0.20

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and non-negative")
            if (key.endswith("fraction") or key == "blur_mix_max") and value >= 1:
                raise ValueError(f"{key} must be below one")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())) if path else None


class EpisodeDomain:
    """Restore nominal arrays before every draw; never consume the task's RNG."""

    ARRAYS = ("body_mass", "body_inertia", "geom_friction", "dof_damping",
              "actuator_gainprm", "actuator_biasprm", "cam_pos", "cam_fovy",
              "light_diffuse", "light_ambient", "mat_rgba", "geom_rgba")

    def __init__(self, model, config: DomainConfig, camera_name="closeup"):
        self.model, self.config = model, config
        self.nominal = {name: getattr(model, name).copy() for name in self.ARRAYS}
        self.cube = model.body("task_cube").id
        self.camera = model.camera(camera_name).id
        if config.camera_fovy_deg >= min(model.cam_fovy[self.camera], 180 - model.cam_fovy[self.camera]):
            raise ValueError("Camera field of view randomization exceeds (0, 180)")
        self.parameters = None

    def restore(self):
        for name, value in self.nominal.items():
            getattr(self.model, name)[:] = value

    def reset(self, seed, data):
        import mujoco

        self.restore()
        physics_seed, visual_seed, sensor_seed = np.random.SeedSequence([int(seed), 73129]).spawn(3)
        physics = np.random.default_rng(physics_seed)
        visual = np.random.default_rng(visual_seed)
        self.sensor = np.random.default_rng(sensor_seed)
        c, m = self.config, self.model
        mass = float(physics.uniform(1 - c.mass_fraction, 1 + c.mass_fraction))
        friction = float(physics.uniform(1 - c.friction_fraction, 1 + c.friction_fraction))
        gain = physics.uniform(1 - c.gain_fraction, 1 + c.gain_fraction, m.nu)
        damping = physics.uniform(1 - c.damping_fraction, 1 + c.damping_fraction, m.nv)
        m.body_mass[self.cube] *= mass
        m.body_inertia[self.cube] *= mass
        m.geom_friction[:] *= friction
        m.dof_damping[:] *= damping
        # Position actuators use gain * target - gain * qpos; scale both terms.
        m.actuator_gainprm[:, 0] *= gain
        m.actuator_biasprm[:, 1] *= gain
        camera_delta = visual.uniform(-c.camera_position_m, c.camera_position_m, 3)
        m.cam_pos[self.camera] += camera_delta
        m.cam_fovy[self.camera] += visual.uniform(-c.camera_fovy_deg, c.camera_fovy_deg)
        light = visual.uniform(1 - c.light_fraction, 1 + c.light_fraction, (m.nlight, 1))
        m.light_diffuse[:] *= light
        m.light_ambient[:] += visual.uniform(0, c.light_fraction * .1, (m.nlight, 1))
        material = visual.uniform(1 - c.material_fraction, 1 + c.material_fraction, (m.nmat, 3))
        geom = visual.uniform(1 - c.material_fraction, 1 + c.material_fraction, (m.ngeom, 1))
        # A non-default geom RGBA overrides the assigned material in MuJoCo.
        geom[m.geom_matid >= 0] = 1
        m.mat_rgba[:, :3] = np.clip(m.mat_rgba[:, :3] * material, 0, 1)
        m.geom_rgba[:, :3] = np.clip(m.geom_rgba[:, :3] * geom, 0, 1)
        self.exposure = float(visual.uniform(1 - c.exposure_fraction, 1 + c.exposure_fraction))
        self.white_balance = visual.uniform(1 - c.white_balance_fraction, 1 + c.white_balance_fraction, 3)
        self.noise_std = float(visual.uniform(0, c.noise_std_max))
        self.blur_mix = float(visual.uniform(0, c.blur_mix_max))
        # Refresh mass-dependent constants and target-camera reference positions before task reset.
        mujoco.mj_setConst(m, data)
        self.parameters = {"seed": int(seed), "mass_scale": mass, "friction_scale": friction,
                           "gain_scale": gain.tolist(), "damping_scale": damping.tolist(),
                           "camera_delta_m": camera_delta.tolist(), "camera_fovy_deg": float(m.cam_fovy[self.camera]),
                           "light_scale": light.tolist(), "material_scale": material.tolist(),
                           "light_ambient": m.light_ambient.tolist(),
                           "geom_color_scale": geom.tolist(), "exposure": self.exposure,
                           "white_balance": self.white_balance.tolist(), "noise_std": self.noise_std,
                           "blur_mix": self.blur_mix}
        return self.parameters

    def image(self, rgb):
        pixels = rgb.astype(np.float32)
        if self.blur_mix:
            padded = np.pad(pixels, ((1, 1), (1, 1), (0, 0)), mode="edge")
            blurred = sum(padded[i:i + len(pixels), j:j + pixels.shape[1]] for i in range(3) for j in range(3)) / 9
            pixels = (1 - self.blur_mix) * pixels + self.blur_mix * blurred
        pixels *= self.exposure * self.white_balance
        if self.noise_std:
            pixels += self.sensor.normal(0, self.noise_std, pixels.shape)
        return np.clip(np.rint(pixels), 0, 255).astype(np.uint8)

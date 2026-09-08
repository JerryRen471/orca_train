from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from orca_sim.orientation import canonicalize_quaternion


class OrcaStateCubeEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Bounded state view of the full-orientation cube task."""

    def __init__(
        self,
        *,
        env: gym.Env | None = None,
        version: str = "v2",
        max_delta_degrees: float = 3.0,
        fixed_joint_names: tuple[str, ...] = (),
        hand_mount_body_name: str = "right_mount",
        cube_body_name: str = "task_cube",
        joint_velocity_limit: float = 10.0,
        workspace_radius: float = 0.25,
        linear_velocity_limit: float = 2.0,
        angular_velocity_limit: float = 20.0,
        **task_kwargs: Any,
    ) -> None:
        super().__init__()
        scales = {
            "max_delta_degrees": max_delta_degrees,
            "joint_velocity_limit": joint_velocity_limit,
            "workspace_radius": workspace_radius,
            "linear_velocity_limit": linear_velocity_limit,
            "angular_velocity_limit": angular_velocity_limit,
        }
        invalid = [
            name for name, value in scales.items()
            if not np.isfinite(value) or value <= 0.0
        ]
        if invalid:
            raise ValueError(f"State environment scales must be positive: {invalid}")

        if env is None:
            from orca_sim import OrcaHandRightCubeOrientation

            task_kwargs.setdefault("goal_mode", "cube_orientation")
            task_kwargs.setdefault("hand_mount_body_name", hand_mount_body_name)
            task_kwargs.setdefault("cube_body_name", cube_body_name)
            env = OrcaHandRightCubeOrientation(version=version, **task_kwargs)

        self.env = env
        self.max_delta_radians = float(np.deg2rad(max_delta_degrees))
        self.fixed_joint_names = frozenset(fixed_joint_names)
        self.joint_velocity_limit = float(joint_velocity_limit)
        self.workspace_radius = float(workspace_radius)
        self.linear_velocity_limit = float(linear_velocity_limit)
        self.angular_velocity_limit = float(angular_velocity_limit)

        model = self.env.model
        joint_ids = np.asarray(model.actuator_trnid[: model.nu, 0], dtype=np.int32)
        self._actuator_qpos_indices = np.asarray(
            model.jnt_qposadr[joint_ids], dtype=np.int32
        )
        self._actuator_qvel_indices = np.asarray(
            model.jnt_dofadr[joint_ids], dtype=np.int32
        )
        if tuple(self.env.action_space.shape) != (17,):
            raise ValueError(
                f"Expected 17 ORCA actuators, got {self.env.action_space.shape}"
            )

        self._joint_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        self._joint_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        joint_names = [model.joint(int(joint_id)).name for joint_id in joint_ids]
        missing = self.fixed_joint_names.difference(joint_names)
        if missing:
            raise ValueError(
                f"Fixed joints are not actuated by this model: {sorted(missing)}"
            )
        self._active_actuator_indices = np.asarray(
            [
                index
                for index, name in enumerate(joint_names)
                if name not in self.fixed_joint_names
            ],
            dtype=np.int32,
        )
        self._hand_mount_body_id = model.body(hand_mount_body_name).id
        self._cube_body_id = model.body(cube_body_name).id

        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(len(self._active_actuator_indices),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {"state": spaces.Box(-1.0, 1.0, shape=(64,), dtype=np.float32)}
        )
        self._target = np.zeros(17, dtype=np.float32)

    def _joint_angles(self) -> np.ndarray:
        return np.asarray(
            self.env.data.qpos[self._actuator_qpos_indices], dtype=np.float32
        ).copy()

    def _joint_velocities(self) -> np.ndarray:
        return np.asarray(
            self.env.data.qvel[self._actuator_qvel_indices], dtype=np.float32
        ).copy()

    def _observation(self, info: dict[str, Any]) -> dict[str, np.ndarray]:
        joint_ranges = self._joint_high - self._joint_low
        if np.any(joint_ranges <= 0.0):
            raise ValueError("Every actuated joint must have a positive ROM")
        normalized_qpos = 2.0 * (
            self._joint_angles() - self._joint_low
        ) / joint_ranges - 1.0
        normalized_qvel = self._joint_velocities() / self.joint_velocity_limit

        mount_position = np.asarray(
            self.env.data.xpos[self._hand_mount_body_id], dtype=np.float64
        )
        mount_rotation = np.asarray(
            self.env.data.xmat[self._hand_mount_body_id], dtype=np.float64
        ).reshape(3, 3)
        cube_rotation = np.asarray(
            self.env.data.xmat[self._cube_body_id], dtype=np.float64
        ).reshape(3, 3)
        cube_position = np.asarray(info["cube_pos"], dtype=np.float64)
        cube_velocity = np.asarray(info["cube_qvel"], dtype=np.float64)
        if cube_position.shape != (3,):
            raise ValueError(f"Expected cube_pos shape (3,), got {cube_position.shape}")
        if cube_velocity.shape != (6,):
            raise ValueError(f"Expected cube_qvel shape (6,), got {cube_velocity.shape}")

        position_in_hand = (
            mount_rotation.T @ (cube_position - mount_position)
        ) / self.workspace_radius
        linear_velocity_in_hand = (
            mount_rotation.T @ cube_velocity[:3]
        ) / self.linear_velocity_limit
        angular_velocity_in_hand = (
            mount_rotation.T @ cube_rotation @ cube_velocity[3:]
        ) / self.angular_velocity_limit
        relative_target = canonicalize_quaternion(
            np.asarray(info["relative_target_quat"], dtype=np.float64)
        )

        state = np.concatenate(
            [
                normalized_qpos,
                normalized_qvel,
                position_in_hand,
                linear_velocity_in_hand,
                angular_velocity_in_hand,
                relative_target,
                2.0 * (self._target - self._joint_low) / joint_ranges - 1.0,
            ]
        )
        return {"state": np.clip(state, -1.0, 1.0).astype(np.float32)}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        _, info = self.env.reset(seed=seed, options=options)
        self._target = np.clip(
            self.env.data.ctrl, self._joint_low, self._joint_high
        ).astype(np.float32).copy()
        return self._observation(info), info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected normalized action shape {self.action_space.shape}, "
                f"got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Action must contain only finite values")
        active = self._active_actuator_indices
        self._target[active] = np.clip(
            self._target[active]
            + np.clip(action, -1.0, 1.0) * self.max_delta_radians,
            self._joint_low[active],
            self._joint_high[active],
        ).astype(np.float32)
        _, reward, terminated, truncated, info = self.env.step(self._target)
        info = dict(info)
        ranges = self._joint_high[active] - self._joint_low[active]
        measured = self._joint_angles()[active]
        for name, positions in (
            ("joint_target_limit_fraction", self._target[active]),
            ("joint_position_limit_fraction", measured),
        ):
            normalized = (positions - self._joint_low[active]) / ranges
            info[name] = float(np.mean((normalized <= 0.01) | (normalized >= 0.99)))
        info["controller_tracking_error_rms_deg"] = float(np.rad2deg(
            np.sqrt(np.mean((self._target[active] - measured) ** 2))
        ))
        return self._observation(info), float(reward), terminated, truncated, info

    def close(self) -> None:
        self.env.close()

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class OrcaVisualCubeEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Pixel/proprioception view of the ORCA cube task with delta actions."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        env: gym.Env | None = None,
        renderer: Any | None = None,
        version: str = "v2",
        image_size: int = 84,
        frame_stack: int = 3,
        max_delta_degrees: float = 3.0,
        camera_name: str = "closeup",
        randomize_reset: bool = True,
        cube_pos_xy_jitter: float = 0.01,
        fixed_joint_names: tuple[str, ...] = (),
        drop_penalty: float = 1.0,
        drop_height: float = 0.10,
        success_height: float = 0.12,
        success_hold_steps: int = 10,
        max_success_linear_speed: float = 0.15,
        max_success_angular_speed: float = 2.0,
        reward_mode: str = "absolute",
        progress_reward_scale: float = 5.0,
        success_bonus: float = 10.0,
        step_penalty: float = 0.01,
    ) -> None:
        super().__init__()
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if frame_stack <= 0:
            raise ValueError("frame_stack must be positive")
        if max_delta_degrees <= 0:
            raise ValueError("max_delta_degrees must be positive")

        if env is None:
            from orca_sim import OrcaHandRightCubeOrientation

            env = OrcaHandRightCubeOrientation(
                version=version,
                render_mode=None,
                initial_red_face="down",
                cube_pos_xy_jitter=cube_pos_xy_jitter,
                drop_penalty=drop_penalty,
                drop_height=drop_height,
                success_height=success_height,
                success_hold_steps=success_hold_steps,
                max_success_linear_speed=max_success_linear_speed,
                max_success_angular_speed=max_success_angular_speed,
                reward_mode=reward_mode,
                progress_reward_scale=progress_reward_scale,
                success_bonus=success_bonus,
                step_penalty=step_penalty,
            )

        self.env = env
        self.image_size = int(image_size)
        self.frame_stack = int(frame_stack)
        self.max_delta_radians = float(np.deg2rad(max_delta_degrees))
        self.camera_name = camera_name
        self.randomize_reset = bool(randomize_reset)
        self.cube_pos_xy_jitter = float(cube_pos_xy_jitter)
        self.fixed_joint_names = frozenset(fixed_joint_names)

        self._actuator_qpos_indices = self._resolve_actuator_qpos_indices()
        action_shape = tuple(self.env.action_space.shape)
        if action_shape != (17,):
            raise ValueError(f"Expected 17 ORCA actuators, got action shape {action_shape}")

        self._joint_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        self._joint_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        joint_names = [
            self.env.model.joint(int(self.env.model.actuator_trnid[index, 0])).name
            for index in range(17)
        ]
        missing = self.fixed_joint_names.difference(joint_names)
        if missing:
            raise ValueError(f"Fixed joints are not actuated by this model: {sorted(missing)}")
        self._active_actuator_indices = np.asarray(
            [index for index, name in enumerate(joint_names) if name not in self.fixed_joint_names],
            dtype=np.int32,
        )
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(len(self._active_actuator_indices),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(
                    0,
                    255,
                    shape=(3 * self.frame_stack, self.image_size, self.image_size),
                    dtype=np.uint8,
                ),
                "proprio": spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32),
            }
        )

        self._renderer = renderer if renderer is not None else self._make_renderer()
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._target = np.zeros(17, dtype=np.float32)

    def _make_renderer(self):
        import mujoco

        return mujoco.Renderer(
            self.env.model,
            height=self.image_size,
            width=self.image_size,
        )

    def _resolve_actuator_qpos_indices(self) -> np.ndarray:
        model = self.env.model
        joint_ids = np.asarray(model.actuator_trnid[: model.nu, 0], dtype=np.int32)
        return np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32)

    def _render_chw(self) -> np.ndarray:
        update_scene = getattr(self._renderer, "update_scene", None)
        if update_scene is not None:
            update_scene(self.env.data, camera=self.camera_name)
        frame = np.asarray(self._renderer.render(), dtype=np.uint8)
        expected = (self.image_size, self.image_size, 3)
        if frame.shape != expected:
            raise RuntimeError(f"Expected rendered frame shape {expected}, got {frame.shape}")
        return np.ascontiguousarray(frame.transpose(2, 0, 1))

    def _joint_angles(self) -> np.ndarray:
        return np.asarray(
            self.env.data.qpos[self._actuator_qpos_indices],
            dtype=np.float32,
        ).copy()

    def _proprio(self) -> np.ndarray:
        joint_angles = self._joint_angles()
        joint_ranges = self._joint_high - self._joint_low
        if np.any(joint_ranges <= 0):
            raise ValueError("Every actuated joint must have a positive ROM")
        normalized = 2.0 * (joint_angles - self._joint_low) / joint_ranges - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "pixels": np.concatenate(tuple(self._frames), axis=0),
            "proprio": self._proprio(),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        reset_options = options
        if reset_options is None and self.randomize_reset and hasattr(
            self.env, "sample_randomized_reset_options"
        ):
            reset_options = self.env.sample_randomized_reset_options(
                seed=seed,
                initial_red_face="down",
                cube_pos_xy_jitter=self.cube_pos_xy_jitter,
            )
        _, info = self.env.reset(seed=seed, options=reset_options)
        self._target = np.clip(self._joint_angles(), self._joint_low, self._joint_high)
        frame = self._render_chw()
        self._frames.clear()
        self._frames.extend(frame.copy() for _ in range(self.frame_stack))
        return self._observation(), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected normalized action shape {self.action_space.shape}, got {action.shape}"
            )
        normalized = np.clip(action, -1.0, 1.0)
        active = self._active_actuator_indices
        measured = self._joint_angles()
        self._target[active] = np.clip(
            measured[active] + normalized * self.max_delta_radians,
            self._joint_low[active],
            self._joint_high[active],
        ).astype(np.float32)
        _, reward, terminated, truncated, info = self.env.step(self._target)
        self._frames.append(self._render_chw())
        return self._observation(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        return np.ascontiguousarray(self._frames[-1].transpose(1, 2, 0))

    def close(self) -> None:
        try:
            self._renderer.close()
        finally:
            self.env.close()

from __future__ import annotations

import numpy as np
import torch

from .state_agent import StateAgent


class GuardedStatePolicy:
    """State-only action projection and target holding for a learned actor.

    Joint ranges and actuator order must match the environment that encoded the
    observation. The hold threshold should be inside the task's success tolerance.
    This guards commanded targets; it cannot guarantee measured joint positions.
    """

    def __init__(
        self, agent: StateAgent, *, joint_low: np.ndarray,
        joint_high: np.ndarray, active_actuator_indices: np.ndarray,
        max_delta_radians: float, joint_target_margin_fraction: float = .05,
        hold_error_degrees: float | None = 10.0,
    ):
        self.agent = agent
        self.action_dim, self.state_dim = agent.action_dim, agent.state_dim
        self.joint_low = np.asarray(joint_low, dtype=np.float64).copy()
        self.joint_high = np.asarray(joint_high, dtype=np.float64).copy()
        if (self.joint_low.ndim != 1 or self.joint_high.shape != self.joint_low.shape
                or not np.all(np.isfinite(self.joint_low))
                or not np.all(np.isfinite(self.joint_high))
                or np.any(self.joint_high <= self.joint_low)):
            raise ValueError('joint ranges must be finite, increasing one-dimensional arrays')
        self.num_joints = self.joint_low.size
        if self.state_dim != 3 * self.num_joints + 13:
            raise ValueError('state dimension does not match the ORCA state observation layout')
        active = np.asarray(active_actuator_indices)
        if (active.shape != (self.action_dim,) or not np.issubdtype(active.dtype, np.integer)
                or np.any(active < 0) or np.any(active >= self.num_joints)
                or np.unique(active).size != active.size):
            raise ValueError('active actuator indices must match the action dimension and joint ranges')
        self.active = active.copy()
        if not np.isfinite(max_delta_radians) or max_delta_radians <= 0:
            raise ValueError('max_delta_radians must be finite and positive')
        if not np.isfinite(joint_target_margin_fraction) or not 0 <= joint_target_margin_fraction < .5:
            raise ValueError('joint target margin must be finite and in [0, 0.5)')
        if hold_error_degrees is not None and (
            not np.isfinite(hold_error_degrees) or not 0 <= hold_error_degrees < 180
        ):
            raise ValueError('hold_error_degrees must be finite and in [0, 180)')
        self.max_delta_radians = float(max_delta_radians)
        self.margin = float(joint_target_margin_fraction)
        self.hold_error_degrees = hold_error_degrees

    def act(self, observation, step, eval_mode=False):
        state = np.asarray(observation['state'], dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f'Expected state shape ({self.state_dim},), got {state.shape}')
        if not np.all(np.isfinite(state)):
            raise ValueError('State observation must contain only finite values')
        quaternion_start = 2 * self.num_joints + 9
        angle = np.rad2deg(2 * np.arccos(np.clip(abs(float(state[quaternion_start])), 0, 1)))
        holding = self.hold_error_degrees is not None and angle <= self.hold_error_degrees
        action = np.zeros(self.action_dim, dtype=np.float32) if holding else np.clip(
            self.agent.act(observation, step, eval_mode=eval_mode), -1, 1)
        ranges = self.joint_high - self.joint_low
        targets = self.joint_low + .5 * (state[-self.num_joints:].astype(np.float64) + 1) * ranges
        active = self.active
        desired = np.clip(
            targets[active] + action * self.max_delta_radians,
            self.joint_low[active] + self.margin * ranges[active],
            self.joint_high[active] - self.margin * ranges[active],
        )
        # An initially outside target may require several bounded control steps.
        return np.clip((desired - targets[active]) / self.max_delta_radians, -1, 1).astype(np.float32)

    @torch.no_grad()
    def value(self, observation):
        action = self.act(observation, step=0, eval_mode=True)
        state = torch.as_tensor(observation['state'], dtype=torch.float32, device=self.agent.device).unsqueeze(0)
        action_tensor = torch.as_tensor(action, device=self.agent.device).unsqueeze(0)
        q1, q2 = self.agent.critic(state, action_tensor)
        return float(torch.minimum(q1, q2).item())

from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from orca_train.state_env import OrcaStateCubeEnv


class _FakeStateCubeEnv(gym.Env):
    MOUNT_POS = np.array([1.0, 2.0, 3.0])
    MOUNT_ROTATION = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    LOCAL_CUBE_POS = np.array([0.05, -0.10, 0.20])
    LOCAL_LINEAR_VELOCITY = np.array([1.0, -2.0, 0.5])
    LOCAL_ANGULAR_VELOCITY = np.array([10.0, -20.0, 5.0])
    RELATIVE_TARGET_QUAT = np.array(
        [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=np.float32
    )

    def __init__(self):
        self.action_space = gym.spaces.Box(
            -0.5, 0.5, shape=(17,), dtype=np.float32
        )
        joint_names = ["right_wrist", *[f"finger_{index}" for index in range(1, 17)]]
        self.model = SimpleNamespace(
            nu=17,
            actuator_trnid=np.column_stack(
                [np.arange(17, dtype=np.int32), np.zeros(17, dtype=np.int32)]
            ),
            jnt_qposadr=np.arange(17, dtype=np.int32),
            jnt_dofadr=np.arange(17, dtype=np.int32),
            joint=lambda joint_id: SimpleNamespace(name=joint_names[joint_id]),
            body=lambda name: SimpleNamespace(id=0),
        )
        self.data = SimpleNamespace(
            qpos=np.linspace(-0.25, 0.25, 17),
            qvel=np.linspace(-10.0, 10.0, 17),
            xpos=np.array([self.MOUNT_POS]),
            xmat=np.array([self.MOUNT_ROTATION.reshape(-1)]),
        )
        self.last_action = None

    def _info(self):
        rotation = self.MOUNT_ROTATION
        return {
            "cube_pos": self.MOUNT_POS + rotation @ self.LOCAL_CUBE_POS,
            "cube_qvel": np.concatenate(
                [
                    rotation @ self.LOCAL_LINEAR_VELOCITY,
                    rotation @ self.LOCAL_ANGULAR_VELOCITY,
                ]
            ),
            "relative_target_quat": self.RELATIVE_TARGET_QUAT.copy(),
        }

    def reset(self, *, seed=None, options=None):
        self.data.qpos[:] = np.linspace(-0.25, 0.25, 17)
        self.data.qvel[:] = np.linspace(-10.0, 10.0, 17)
        return np.zeros(58), self._info()

    def step(self, action):
        self.last_action = np.asarray(action).copy()
        self.data.qpos[:] = self.last_action
        return np.zeros(58), 0.75, False, False, self._info()

    def close(self):
        pass


def test_state_observation_has_documented_order_scaling_and_bounds() -> None:
    base = _FakeStateCubeEnv()
    env = OrcaStateCubeEnv(
        env=base,
        joint_velocity_limit=10.0,
        workspace_radius=0.25,
        linear_velocity_limit=2.0,
        angular_velocity_limit=20.0,
    )

    observation, _ = env.reset(seed=7)
    state = observation["state"]

    assert state.shape == (47,)
    assert state.dtype == np.float32
    assert env.observation_space.contains(observation)
    np.testing.assert_allclose(state[0:17], np.linspace(-0.5, 0.5, 17))
    np.testing.assert_allclose(state[17:34], np.linspace(-1.0, 1.0, 17))
    np.testing.assert_allclose(state[34:37], [0.2, -0.4, 0.8])
    np.testing.assert_allclose(state[37:40], [0.5, -1.0, 0.25])
    np.testing.assert_allclose(state[40:43], [0.5, -1.0, 0.25])
    np.testing.assert_allclose(state[43:47], _FakeStateCubeEnv.RELATIVE_TARGET_QUAT)


def test_state_actions_are_measured_deltas_and_can_lock_wrist() -> None:
    base = _FakeStateCubeEnv()
    env = OrcaStateCubeEnv(
        env=base,
        max_delta_degrees=3.0,
        fixed_joint_names=("right_wrist",),
    )
    env.reset()
    wrist_before = base.data.qpos[0]

    env.step(np.ones(16, dtype=np.float32))

    assert env.action_space.shape == (16,)
    assert base.last_action[0] == wrist_before
    expected = np.clip(
        np.linspace(-0.25, 0.25, 17)[1:] + np.deg2rad(3.0),
        base.action_space.low[1:],
        base.action_space.high[1:],
    )
    np.testing.assert_allclose(base.last_action[1:], expected, atol=1e-7)

    measured = np.linspace(0.1, -0.1, 17)
    base.data.qpos[:] = measured
    env.step(np.ones(16, dtype=np.float32))
    expected = np.clip(
        measured[1:] + np.deg2rad(3.0),
        base.action_space.low[1:],
        base.action_space.high[1:],
    )
    np.testing.assert_allclose(base.last_action[1:], expected, atol=1e-7)

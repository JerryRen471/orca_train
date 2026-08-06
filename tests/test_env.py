from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from orca_train.env import OrcaVisualCubeEnv


class _FakeRenderer:
    def __init__(self):
        self.value = 10

    def render(self):
        frame = np.full((84, 84, 3), self.value, dtype=np.uint8)
        self.value += 10
        return frame

    def close(self):
        pass


class _FakeCubeEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-0.5, 0.5, shape=(17,), dtype=np.float32)
        self.model = SimpleNamespace(
            nu=17,
            actuator_trnid=np.column_stack(
                [np.arange(17, dtype=np.int32), np.zeros(17, dtype=np.int32)]
            ),
            jnt_qposadr=np.arange(17, dtype=np.int32),
        )
        self.data = SimpleNamespace(qpos=np.linspace(-0.2, 0.2, 17, dtype=np.float64))
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        self.data.qpos[:] = np.linspace(-0.2, 0.2, 17)
        return np.zeros(51), {"is_success": False}

    def step(self, action):
        self.last_action = np.asarray(action).copy()
        self.data.qpos[:] = self.last_action
        return np.zeros(51), 0.75, False, False, {"is_success": False}

    def close(self):
        pass


def test_visual_env_reset_returns_three_identical_rgb_frames_and_joint_angles():
    base = _FakeCubeEnv()
    env = OrcaVisualCubeEnv(env=base, renderer=_FakeRenderer())

    obs, info = env.reset(seed=1)

    assert obs["pixels"].shape == (9, 84, 84)
    assert obs["pixels"].dtype == np.uint8
    np.testing.assert_array_equal(obs["pixels"][:3], obs["pixels"][3:6])
    np.testing.assert_array_equal(obs["pixels"][:3], obs["pixels"][6:])
    np.testing.assert_allclose(obs["proprio"], np.linspace(-0.2, 0.2, 17))
    assert info == {"is_success": False}


def test_visual_env_step_stacks_new_frame_and_preserves_task_reward():
    env = OrcaVisualCubeEnv(env=_FakeCubeEnv(), renderer=_FakeRenderer())
    first, _ = env.reset()

    obs, reward, terminated, truncated, info = env.step(np.zeros(17, dtype=np.float32))

    np.testing.assert_array_equal(obs["pixels"][:6], first["pixels"][3:])
    assert np.all(obs["pixels"][6:] == 20)
    assert reward == 0.75
    assert not terminated
    assert not truncated
    assert info == {"is_success": False}


def test_visual_env_action_is_three_degree_increment_and_clipped_to_joint_rom():
    base = _FakeCubeEnv()
    env = OrcaVisualCubeEnv(env=base, renderer=_FakeRenderer(), max_delta_degrees=3.0)
    env.reset()

    env.step(np.ones(17, dtype=np.float32))
    expected = np.clip(
        np.linspace(-0.2, 0.2, 17) + np.deg2rad(3.0),
        base.action_space.low,
        base.action_space.high,
    )
    np.testing.assert_allclose(base.last_action, expected, atol=1e-7)
    assert env.action_space.low.tolist() == [-1.0] * 17
    assert env.action_space.high.tolist() == [1.0] * 17

import numpy as np
import torch

from orca_train.visualize import infer_action_dim, run_episode


class FakeAgent:
    def __init__(self):
        self.calls = []

    def act(self, observation, step, eval_mode):
        self.calls.append((step, eval_mode))
        return np.zeros(17, dtype=np.float32)


class FakeEnv:
    def __init__(self):
        self.steps = 0

    def reset(self, seed=None):
        self.steps = 0
        return {"pixels": np.zeros((9, 84, 84)), "proprio": np.zeros(17)}, {}

    def step(self, action):
        self.steps += 1
        info = {"is_success": self.steps == 2, "dropped": False}
        return {}, 0.5, self.steps == 2, False, info


class FakeViewer:
    def __init__(self):
        self.sync_count = 0

    def is_running(self):
        return True

    def sync(self):
        self.sync_count += 1


def test_run_episode_uses_deterministic_policy_and_syncs_viewer():
    agent = FakeAgent()
    env = FakeEnv()
    viewer = FakeViewer()

    result = run_episode(env, agent, viewer, seed=7, realtime=False)

    assert result == {
        "seed": 7,
        "return": 1.0,
        "length": 2,
        "success": True,
        "dropped": False,
    }
    assert agent.calls == [(0, True), (0, True)]
    assert viewer.sync_count == 3


def test_infer_action_dim_reads_actor_output_layer():
    state = {"actor": {"policy.4.weight": torch.zeros(16, 32)}}

    assert infer_action_dim(state) == 16

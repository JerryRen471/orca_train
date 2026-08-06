import json

import numpy as np

from orca_train.train import TrainConfig, train


class TinyEnv:
    def __init__(self):
        self.steps = 0
        self.closed = False

    @staticmethod
    def observation(value=0):
        return {
            "pixels": np.full((9, 84, 84), value, dtype=np.uint8),
            "proprio": np.zeros(17, dtype=np.float32),
        }

    def reset(self, seed=None):
        self.steps = 0
        return self.observation(), {}

    def step(self, action):
        self.steps += 1
        terminated = self.steps == 2
        return self.observation(self.steps), 1.0, terminated, False, {"success": terminated}

    def close(self):
        self.closed = True


def test_training_loop_logs_and_checkpoints(tmp_path):
    environments = []

    def factory():
        env = TinyEnv()
        environments.append(env)
        return env

    config = TrainConfig(
        total_steps=4,
        seed_steps=4,
        eval_every_steps=2,
        eval_episodes=1,
        checkpoint_every_steps=2,
        replay_capacity=16,
        output_dir=tmp_path,
        device="cpu",
        feature_dim=8,
        hidden_dim=16,
    )

    train(config, env_factory=factory)

    assert (tmp_path / "checkpoint_4.pt").exists()
    events = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert any(event["type"] == "train_episode" for event in events)
    assert any(event["type"] == "evaluation" for event in events)
    assert all(env.closed for env in environments)

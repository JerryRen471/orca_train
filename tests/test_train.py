import json

import numpy as np
from gymnasium import spaces

from orca_train.train import TrainConfig, evaluate, train


class TinyEnv:
    def __init__(self):
        self.steps = 0
        self.closed = False
        self.action_space = spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32)
        self.action_shapes = []

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
        self.action_shapes.append(action.shape)
        self.steps += 1
        terminated = self.steps == 2
        return self.observation(self.steps), 1.0, terminated, False, {
            "success": terminated,
            "dropped": not terminated,
        }

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
    assert all(shape == (16,) for env in environments for shape in env.action_shapes)


class EvaluationEnv(TinyEnv):
    def __init__(self):
        super().__init__()
        self.reset_seeds = []

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return super().reset(seed=seed)


class ZeroAgent:
    def act(self, observation, step, eval_mode=False):
        return np.zeros(16, dtype=np.float32)


def test_evaluation_uses_fixed_seed_set_and_reports_drop_rate():
    env = EvaluationEnv()
    result = evaluate(ZeroAgent(), env, episodes=3, seed=100)

    assert env.reset_seeds == [100, 101, 102]
    assert result == {"return": 2.0, "success_rate": 1.0, "drop_rate": 1.0}


def test_corrected_training_defaults():
    config = TrainConfig()
    assert config.replay_capacity == 200_000
    assert config.nstep == 3
    assert config.eval_episodes == 20
    assert config.eval_seed == 10_000
    assert config.reward_mode == "progress"
    assert config.drop_penalty == 10.0

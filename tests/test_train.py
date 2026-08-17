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
            "is_success": terminated,
            "dropped": not terminated,
            "target_completed": terminated,
            "completed_target_steps": 2 if terminated else None,
            "tasks_completed": 1 if terminated else 0,
            "termination_reason": "sequence_complete" if terminated else None,
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
    train_episode = next(event for event in events if event["type"] == "train_episode")
    assert train_episode["tasks_completed"] == 1
    assert train_episode["termination_reason"] == "sequence_complete"
    assert all(env.closed for env in environments)
    assert all(shape == (16,) for env in environments for shape in env.action_shapes)


class EvaluationEnv(TinyEnv):
    def __init__(self):
        super().__init__()
        self.reset_seeds = []

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return super().reset(seed=seed)

    def step(self, action):
        self.steps += 1
        seed = self.reset_seeds[-1]
        info = {
            "is_success": False,
            "dropped": False,
            "target_completed": False,
            "completed_target_steps": None,
            "tasks_completed": 0,
            "termination_reason": None,
        }
        terminated = False
        if seed == 100:
            info.update(
                target_completed=True,
                completed_target_steps=4 if self.steps == 1 else 6,
                tasks_completed=self.steps,
            )
            terminated = self.steps == 2
            if terminated:
                info.update(is_success=True, termination_reason="sequence_complete")
        elif seed == 101:
            if self.steps == 1:
                info.update(
                    target_completed=True,
                    completed_target_steps=5,
                    tasks_completed=1,
                )
            else:
                info.update(dropped=True, tasks_completed=1, termination_reason="dropped")
                terminated = True
        else:
            info.update(termination_reason="task_timeout")
            terminated = True
        return self.observation(self.steps), 1.0, terminated, False, info


class ZeroAgent:
    def act(self, observation, step, eval_mode=False):
        return np.zeros(16, dtype=np.float32)


def test_evaluation_reports_multi_target_sequence_metrics():
    env = EvaluationEnv()
    result = evaluate(ZeroAgent(), env, episodes=3, seed=100)

    assert env.reset_seeds == [100, 101, 102]
    assert result == {
        "return": 5.0 / 3.0,
        "success_rate": 1.0 / 3.0,
        "mean_tasks_completed": 1.0,
        "total_tasks_completed": 3,
        "drop_rate": 1.0 / 3.0,
        "timeout_rate": 1.0 / 3.0,
        "mean_steps_per_completed_target": 5.0,
    }


def test_corrected_training_defaults():
    config = TrainConfig()
    assert config.replay_capacity == 200_000
    assert config.nstep == 3
    assert config.eval_episodes == 20
    assert config.eval_seed == 10_000
    assert config.reward_mode == "progress"
    assert config.drop_penalty == 10.0
    assert config.drop_height == 0.10
    assert config.success_height == 0.12
    assert config.success_hold_steps == 10
    assert config.max_success_linear_speed == 0.15
    assert config.max_success_angular_speed == 2.0
    assert config.target_sequence_length == 20
    assert config.max_task_steps == 200

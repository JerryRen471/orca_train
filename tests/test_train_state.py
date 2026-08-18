import json
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

from orca_train.train_state import (
    StateTrainConfig,
    evaluate_state,
    parse_args,
    resolve_preset,
    train_state,
)
from orca_train.state_agent import StateAgent, StateAgentConfig


@pytest.mark.parametrize(
    ("preset", "target_policy", "sequence_length", "tolerance_rad"),
    [
        ("single_goal", "fixed_quarter_turn", 1, 0.4),
        ("right_angle", "random_quarter_turn", 1, 0.4),
        ("multi_goal", "cube_orientation_bag", 20, np.deg2rad(15.0)),
    ],
)
def test_state_curriculum_presets_resolve_exact_task_settings(
    preset, target_policy, sequence_length, tolerance_rad
) -> None:
    resolved = resolve_preset(StateTrainConfig(preset=preset))

    assert resolved.target_policy == target_policy
    assert resolved.target_sequence_length == sequence_length
    assert resolved.success_tolerance_rad == pytest.approx(tolerance_rad)
    assert resolved.control_period_s == pytest.approx(0.08)
    assert resolved.max_task_duration_s == pytest.approx(8.0)
    assert resolved.success_hold_duration_s == pytest.approx(0.16)
    assert resolved.reward_mode == "angular_progress"


def test_state_curriculum_rejects_unknown_preset() -> None:
    with pytest.raises(ValueError, match="Unknown state curriculum preset"):
        resolve_preset(StateTrainConfig(preset="unknown"))


def test_state_curriculum_derives_distinct_default_run_directories() -> None:
    resolved = {
        preset: resolve_preset(StateTrainConfig(preset=preset, seed=7)).output_dir
        for preset in ("single_goal", "right_angle", "multi_goal")
    }

    assert resolved == {
        "single_goal": Path("runs/state_cube_single_goal_seed7"),
        "right_angle": Path("runs/state_cube_right_angle_seed7"),
        "multi_goal": Path("runs/state_cube_multi_goal_seed7"),
    }


def test_explicit_task_values_override_curriculum_defaults() -> None:
    resolved = resolve_preset(
        StateTrainConfig(
            preset="multi_goal",
            target_policy="fixed_quarter_turn",
            target_sequence_length=3,
            success_tolerance_rad=0.2,
            control_period_s=0.04,
            max_task_duration_s=4.0,
            success_hold_duration_s=0.12,
            reward_mode="angular_progress",
        )
    )

    assert resolved.target_policy == "fixed_quarter_turn"
    assert resolved.target_sequence_length == 3
    assert resolved.success_tolerance_rad == pytest.approx(0.2)
    assert resolved.control_period_s == pytest.approx(0.04)
    assert resolved.max_task_duration_s == pytest.approx(4.0)
    assert resolved.success_hold_duration_s == pytest.approx(0.12)


class _ZeroAgent:
    def act(self, observation, step, eval_mode=False):
        return np.zeros(16, dtype=np.float32)


class _EvaluationStateEnv:
    def __init__(self):
        self.action_space = spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32)
        self.steps = 0
        self.seed = 0
        self.reset_seeds = []

    @staticmethod
    def _observation():
        return {"state": np.zeros(47, dtype=np.float32)}

    def reset(self, seed=None):
        self.seed = seed
        self.steps = 0
        self.reset_seeds.append(seed)
        reset = {
            100: (np.pi / 2.0, 1.0),
            101: (np.pi, 1.2),
            102: (2.0 * np.pi / 3.0, 1.0),
        }
        angle, error = reset[seed]
        return self._observation(), {
            "target_rotation_angle_rad": angle,
            "orientation_error_rad": error,
        }

    def step(self, action):
        self.steps += 1
        info = {
            "is_success": False,
            "dropped": False,
            "target_completed": False,
            "completed_target_steps": None,
            "completed_target_rotation_angle_rad": None,
            "tasks_completed": 0,
            "termination_reason": None,
            "control_period_s": 0.08,
        }
        terminated = False
        if self.seed == 100 and self.steps == 1:
            info.update(
                target_completed=True,
                completed_target_steps=2,
                completed_target_rotation_angle_rad=np.pi / 2.0,
                target_rotation_angle_rad=2.0 * np.pi / 3.0,
                orientation_error_rad=0.4,
                tasks_completed=1,
            )
        elif self.seed == 100:
            info.update(
                is_success=True,
                target_completed=True,
                completed_target_steps=3,
                completed_target_rotation_angle_rad=2.0 * np.pi / 3.0,
                target_rotation_angle_rad=2.0 * np.pi / 3.0,
                orientation_error_rad=0.1,
                tasks_completed=2,
                termination_reason="sequence_complete",
            )
            terminated = True
        elif self.seed == 101 and self.steps == 1:
            info.update(
                target_completed=True,
                completed_target_steps=4,
                completed_target_rotation_angle_rad=np.pi,
                target_rotation_angle_rad=np.pi / 2.0,
                orientation_error_rad=0.5,
                tasks_completed=1,
            )
        elif self.seed == 101:
            info.update(
                dropped=True,
                target_rotation_angle_rad=np.pi / 2.0,
                orientation_error_rad=1.0,
                tasks_completed=1,
                termination_reason="dropped",
            )
            terminated = True
        else:
            info.update(
                target_rotation_angle_rad=2.0 * np.pi / 3.0,
                orientation_error_rad=0.8,
                termination_reason="task_timeout",
            )
            terminated = True
        return self._observation(), 1.0, terminated, False, info


def test_state_evaluation_reports_orientation_and_rotation_bucket_metrics() -> None:
    env = _EvaluationStateEnv()

    result = evaluate_state(_ZeroAgent(), env, episodes=3, seed=100)

    assert env.reset_seeds == [100, 101, 102]
    assert result["return"] == pytest.approx(5.0 / 3.0)
    assert result["success_rate"] == pytest.approx(1.0 / 3.0)
    assert result["mean_targets_completed"] == pytest.approx(1.0)
    assert result["median_targets_completed"] == pytest.approx(1.0)
    assert result["total_targets_completed"] == 3
    assert result["drop_rate"] == pytest.approx(1.0 / 3.0)
    assert result["timeout_rate"] == pytest.approx(1.0 / 3.0)
    assert result["mean_seconds_per_completed_target"] == pytest.approx(0.24)
    assert result["median_seconds_per_completed_target"] == pytest.approx(0.24)
    assert result["mean_final_orientation_error_rad"] == pytest.approx(1.9 / 3.0)
    assert result["mean_best_orientation_error_rad"] == pytest.approx(1.4 / 3.0)
    assert result["rotation_buckets"] == {
        "90": {
            "attempts": 2,
            "completed": 1,
            "completion_rate": 0.5,
            "mean_completion_s": pytest.approx(0.16),
        },
        "120": {
            "attempts": 2,
            "completed": 1,
            "completion_rate": 0.5,
            "mean_completion_s": pytest.approx(0.24),
        },
        "180": {
            "attempts": 1,
            "completed": 1,
            "completion_rate": 1.0,
            "mean_completion_s": pytest.approx(0.32),
        },
    }


class _TinyStateEnv:
    def __init__(self):
        self.action_space = spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32)
        self.steps = 0
        self.closed = False
        self.action_shapes = []

    @staticmethod
    def _observation(value=0.0):
        return {"state": np.full(47, value, dtype=np.float32)}

    def reset(self, seed=None):
        self.steps = 0
        return self._observation(), {
            "target_rotation_angle_rad": np.pi / 2.0,
            "orientation_error_rad": 1.0,
        }

    def step(self, action):
        self.action_shapes.append(action.shape)
        self.steps += 1
        terminated = self.steps == 2
        info = {
            "is_success": terminated,
            "dropped": False,
            "target_completed": terminated,
            "completed_target_steps": 2 if terminated else None,
            "completed_target_rotation_angle_rad": (
                np.pi / 2.0 if terminated else None
            ),
            "target_rotation_angle_rad": np.pi / 2.0,
            "orientation_error_rad": 0.1 if terminated else 0.8,
            "control_period_s": 0.08,
            "tasks_completed": 1 if terminated else 0,
            "termination_reason": "sequence_complete" if terminated else None,
        }
        return self._observation(self.steps / 10.0), 1.0, terminated, False, info

    def close(self):
        self.closed = True


def test_state_training_loop_logs_updates_evaluation_and_checkpoints(tmp_path) -> None:
    environments = []

    def factory():
        env = _TinyStateEnv()
        environments.append(env)
        return env

    config = StateTrainConfig(
        total_steps=4,
        seed_steps=2,
        eval_every_steps=2,
        eval_episodes=1,
        checkpoint_every_steps=2,
        replay_capacity=16,
        batch_size=2,
        output_dir=tmp_path,
        device="cpu",
        hidden_dim=32,
    )

    train_state(config, env_factory=factory)

    assert (tmp_path / "checkpoint_4.pt").exists()
    restored = StateAgent(47, 16, "cpu", StateAgentConfig(hidden_dim=32))
    assert restored.load(tmp_path / "checkpoint_4.pt") == 4
    saved_config = json.loads((tmp_path / "config.json").read_text())
    assert saved_config["target_policy"] == "fixed_quarter_turn"
    events = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert any(event["type"] == "update" for event in events)
    assert any(event["type"] == "evaluation" for event in events)
    assert any(event["type"] == "train_episode" for event in events)
    assert all(env.closed for env in environments)
    assert all(
        shape == (16,) for env in environments for shape in env.action_shapes
    )


def test_state_cli_parses_explicit_curriculum_overrides(tmp_path) -> None:
    config = parse_args(
        [
            "--preset",
            "multi_goal",
            "--target-policy",
            "fixed_quarter_turn",
            "--target-sequence-length",
            "3",
            "--success-tolerance-degrees",
            "30",
            "--control-period-s",
            "0.04",
            "--max-task-duration-s",
            "4.0",
            "--success-hold-duration-s",
            "0.12",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert config.preset == "multi_goal"
    assert config.target_policy == "fixed_quarter_turn"
    assert config.target_sequence_length == 3
    assert config.success_tolerance_rad == pytest.approx(np.pi / 6.0)
    assert config.control_period_s == pytest.approx(0.04)
    assert config.max_task_duration_s == pytest.approx(4.0)
    assert config.success_hold_duration_s == pytest.approx(0.12)
    assert config.output_dir == tmp_path


def test_state_cli_rejects_both_tolerance_units() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--success-tolerance-rad",
                "0.4",
                "--success-tolerance-degrees",
                "15",
            ]
        )

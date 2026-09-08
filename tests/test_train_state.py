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
from orca_train import train_state as training


@pytest.mark.parametrize(
    (
        "preset",
        "target_policy",
        "sequence_length",
        "target_angle_rad",
        "tolerance_rad",
    ),
    [
        ("turn_30", "fixed_quarter_turn", 1, np.deg2rad(30.0), np.deg2rad(15.0)),
        ("turn_45", "fixed_quarter_turn", 1, np.deg2rad(45.0), np.deg2rad(15.0)),
        ("turn_60", "fixed_quarter_turn", 1, np.deg2rad(60.0), np.deg2rad(15.0)),
        ("single_goal", "fixed_quarter_turn", 1, np.pi / 2.0, 0.4),
        ("right_angle", "random_quarter_turn", 1, np.pi / 2.0, 0.4),
        (
            "multi_goal",
            "cube_orientation_bag",
            20,
            np.pi / 2.0,
            np.deg2rad(15.0),
        ),
    ],
)
def test_state_curriculum_presets_resolve_exact_task_settings(
    preset, target_policy, sequence_length, target_angle_rad, tolerance_rad
) -> None:
    resolved = resolve_preset(StateTrainConfig(preset=preset))

    assert resolved.target_policy == target_policy
    assert resolved.target_sequence_length == sequence_length
    assert resolved.target_rotation_angle_rad == pytest.approx(target_angle_rad)
    assert resolved.success_tolerance_rad == pytest.approx(tolerance_rad)
    assert resolved.control_period_s == pytest.approx(0.08)
    assert resolved.max_task_duration_s == pytest.approx(8.0)
    assert resolved.success_hold_duration_s == pytest.approx(2.0)
    assert resolved.reward_mode == "angular_progress"


def test_state_curriculum_rejects_unknown_preset() -> None:
    with pytest.raises(ValueError, match="Unknown state curriculum preset"):
        resolve_preset(StateTrainConfig(preset="unknown"))


@pytest.mark.parametrize("scale", [-0.1, 1.1, np.inf, np.nan])
def test_state_training_rejects_invalid_seed_action_scale(scale) -> None:
    with pytest.raises(ValueError, match="seed_action_scale"):
        StateTrainConfig(seed_action_scale=scale)


@pytest.mark.parametrize("clip", [-0.1, np.inf, np.nan])
def test_state_training_rejects_invalid_behavior_noise_clip(clip) -> None:
    with pytest.raises(ValueError, match="behavior_stddev_clip"):
        StateTrainConfig(behavior_stddev_clip=clip)


def test_state_training_rejects_invalid_behavior_noise_schedule() -> None:
    with pytest.raises(ValueError, match="behavior_stddev_schedule"):
        StateTrainConfig(behavior_stddev_schedule="linear(0.2,0.05,0)")


def test_state_curriculum_derives_distinct_default_run_directories() -> None:
    resolved = {
        preset: resolve_preset(StateTrainConfig(preset=preset, seed=7)).output_dir
        for preset in (
            "turn_30",
            "turn_45",
            "turn_60",
            "single_goal",
            "right_angle",
            "multi_goal",
        )
    }

    assert resolved == {
        "turn_30": Path("runs/state_cube_turn_30_seed7"),
        "turn_45": Path("runs/state_cube_turn_45_seed7"),
        "turn_60": Path("runs/state_cube_turn_60_seed7"),
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
            target_rotation_angle_rad=0.7,
            success_tolerance_rad=0.2,
            control_period_s=0.04,
            max_task_duration_s=4.0,
            success_hold_duration_s=0.12,
            reward_mode="angular_progress",
        )
    )

    assert resolved.target_policy == "fixed_quarter_turn"
    assert resolved.target_sequence_length == 3
    assert resolved.target_rotation_angle_rad == pytest.approx(0.7)
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
        observation = self._observation()
        observation["state"][0] = seed / 1000.0
        return observation, {
            "target_rotation_angle_rad": angle,
            "orientation_error_rad": error,
            "success_tolerance_rad": 0.4,
            "cube_pos": np.array([0.0, 0.0, 0.14]),
            "cube_linear_speed": 0.2,
            "cube_angular_speed": 2.5,
            "cube_hand_contact_count": 1,
            "stable_success_steps": 0,
        }

    def step(self, action):
        self.steps += 1
        info = {
            "is_success": False,
            "dropped": False,
            "target_completed": False,
            "completed_target_steps": None,
            "completed_target_rotation_angle_rad": None,
            "completed_target_orientation_error_rad": None,
            "tasks_completed": 0,
            "termination_reason": None,
            "control_period_s": 0.08,
            "success_tolerance_rad": 0.4,
            "cube_pos": np.array([0.0, 0.0, 0.14]),
            "cube_linear_speed": 0.1,
            "cube_angular_speed": 1.5,
            "cube_hand_contact_count": 2,
            "stable_success_steps": 0,
            "success_hold_steps": 2,
        }
        terminated = False
        if self.seed == 100 and self.steps == 1:
            info.update(
                target_completed=True,
                completed_target_steps=2,
                completed_target_rotation_angle_rad=np.pi / 2.0,
                target_rotation_angle_rad=2.0 * np.pi / 3.0,
                orientation_error_rad=0.4,
                completed_target_orientation_error_rad=0.4,
                tasks_completed=1,
                stable_success_steps=1,
            )
        elif self.seed == 100:
            info.update(
                is_success=True,
                target_completed=True,
                completed_target_steps=3,
                completed_target_rotation_angle_rad=2.0 * np.pi / 3.0,
                target_rotation_angle_rad=2.0 * np.pi / 3.0,
                orientation_error_rad=0.1,
                completed_target_orientation_error_rad=0.1,
                tasks_completed=2,
                termination_reason="sequence_complete",
                stable_success_steps=2,
            )
            terminated = True
        elif self.seed == 101 and self.steps == 1:
            info.update(
                target_completed=True,
                completed_target_steps=4,
                completed_target_rotation_angle_rad=np.pi,
                target_rotation_angle_rad=np.pi / 2.0,
                orientation_error_rad=0.5,
                completed_target_orientation_error_rad=0.2,
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
    assert result["success_rate_ci95"]["lower"] <= result["success_rate"]
    assert result["success_rate_ci95"]["upper"] >= result["success_rate"]
    assert result["mean_targets_completed"] == pytest.approx(1.0)
    assert result["median_targets_completed"] == pytest.approx(1.0)
    assert result["total_targets_completed"] == 3
    assert result["drop_rate"] == pytest.approx(1.0 / 3.0)
    assert result["timeout_rate"] == pytest.approx(1.0 / 3.0)
    assert result["mean_seconds_per_completed_target"] == pytest.approx(0.24)
    assert result["median_seconds_per_completed_target"] == pytest.approx(0.24)
    assert result["mean_final_orientation_error_rad"] == pytest.approx(1.9 / 3.0)
    assert result["mean_best_orientation_error_rad"] == pytest.approx(1.1 / 3.0)
    assert result["funnel"] == {
        "reached_error_60deg_rate": 1.0,
        "reached_error_45deg_rate": pytest.approx(2.0 / 3.0),
        "reached_error_30deg_rate": pytest.approx(2.0 / 3.0),
        "reached_success_tolerance_rate": pytest.approx(2.0 / 3.0),
        "mean_max_stable_success_steps": pytest.approx(4.0 / 3.0),
        "mean_drop_step": pytest.approx(2.0),
        "mean_success_tolerance_cube_height_m": pytest.approx(0.14),
        "mean_success_tolerance_linear_speed": pytest.approx(0.1),
        "mean_success_tolerance_angular_speed": pytest.approx(1.5),
        "mean_success_tolerance_hand_contact_count": pytest.approx(2.0),
    }
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


@pytest.mark.parametrize("truncated", [False, True])
def test_evaluation_calibrates_value_only_on_terminal_returns(truncated):
    class KnownAgent:
        def act(self, observation, step, eval_mode=False):
            return np.tile([1.0, 0.0], 8)

        def value(self, observation):
            return 5.0

    class KnownEnv(_EvaluationStateEnv):
        def step(self, action):
            obs, reward, terminal, _, info = super().step(action)
            info["joint_target_limit_fraction"] = 0.25 if self.steps == 1 else 0.75
            return obs, reward, terminal and not truncated, terminal and truncated, info

    result = evaluate_state(KnownAgent(), KnownEnv(), episodes=1, seed=100, gamma=0.5)
    assert result["mean_initial_q"] == 5.0
    assert result["mean_discounted_return"] == 1.5
    assert result["value_calibration_episodes"] == (0 if truncated else 1)
    assert result["mean_value_bias_terminal"] == (None if truncated else 3.5)
    assert result["action_rms"] == pytest.approx(np.sqrt(0.5))
    assert result["action_saturation_fraction"] == 0.5
    assert result["mean_joint_target_limit_fraction"] == 0.5


class _TinyStateEnv:
    def __init__(self):
        self.action_space = spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32)
        self.steps = 0
        self.closed = False
        self.action_shapes = []
        self.actions = []

    @staticmethod
    def _observation(value=0.0):
        return {"state": np.full(47, value, dtype=np.float32)}

    def reset(self, seed=None):
        self.steps = 0
        return self._observation(), {
            "target_rotation_angle_rad": np.pi / 2.0,
            "orientation_error_rad": 1.0,
            "success_tolerance_rad": 0.4,
            "cube_pos": np.array([0.0, 0.0, 0.14]),
            "cube_linear_speed": 0.0,
            "cube_angular_speed": 0.0,
            "cube_hand_contact_count": 2,
            "stable_success_steps": 0,
        }

    def step(self, action):
        self.action_shapes.append(action.shape)
        self.actions.append(np.array(action, copy=True))
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
            "success_tolerance_rad": 0.4,
            "cube_pos": np.array([0.0, 0.0, 0.14]),
            "cube_linear_speed": 0.0,
            "cube_angular_speed": 0.0,
            "cube_hand_contact_count": 2,
            "stable_success_steps": self.steps,
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
    train_events = [event for event in events if event["type"] == "train_episode"]
    evaluation_events = [event for event in events if event["type"] == "evaluation"]
    assert all(
        event["funnel"]["mean_drop_step"] is None
        for event in evaluation_events
    )
    assert all(event["reached_error_60deg"] for event in train_events)
    assert all(event["reached_success_tolerance"] for event in train_events)
    assert all(event["max_stable_success_steps"] == 2 for event in train_events)
    assert all(
        event["success_tolerance_hand_contact_count"] == 2
        for event in train_events
    )
    assert all(env.closed for env in environments)
    assert all(
        shape == (16,) for env in environments for shape in env.action_shapes
    )
    assert all(
        np.max(np.abs(action)) <= 0.1
        for action in environments[0].actions[: config.seed_steps]
    )


def test_state_training_exploration_defaults_match_survival_first_policy() -> None:
    config = StateTrainConfig()

    assert config.eval_episodes == 100
    assert config.seed_action_scale == pytest.approx(0.1)
    assert config.behavior_stddev_schedule == "linear(0.2,0.05,100000)"
    assert config.behavior_stddev_clip == pytest.approx(0.2)
    assert config.gate_orientation_progress_on_grasp
    assert config.grasp_height_reward_scale == pytest.approx(0.01)


def test_state_cli_parses_explicit_curriculum_overrides(tmp_path) -> None:
    config = parse_args(
        [
            "--preset",
            "multi_goal",
            "--target-policy",
            "fixed_quarter_turn",
            "--target-sequence-length",
            "3",
            "--target-rotation-degrees",
            "45",
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
    assert config.target_rotation_angle_rad == pytest.approx(np.pi / 4.0)
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


@pytest.mark.parametrize("preset", ["turn_30", "turn_45", "turn_60"])
def test_corrected_curriculum_keeps_support_but_zero_action_cannot_solve(preset) -> None:
    env = training.make_state_env(resolve_preset(StateTrainConfig(preset=preset)))
    try:
        observations = []
        for seed in (10000, 10001, 10002):
            obs, info = env.reset(seed=seed)
            observations.append(obs["state"])
            assert obs["state"].shape == (64,)
            assert env.action_space.shape == (16,)
            assert info["orientation_error_rad"] > info["success_tolerance_rad"]
            assert info["cube_pos"][2] > 0.12
            for _ in range(100):
                _, _, terminated, truncated, info = env.step(np.zeros(16, dtype=np.float32))
                assert not info["dropped"]
                assert not info["is_success"]
                if terminated or truncated:
                    break
            assert info["termination_reason"] == "task_timeout"
        assert not np.array_equal(observations[0], observations[1])
        repeated, _ = env.reset(seed=10000)
        np.testing.assert_array_equal(observations[0], repeated["state"])
    finally:
        env.close()


def test_evaluation_exposes_duplicate_initial_states() -> None:
    result = evaluate_state(_ZeroAgent(), _TinyStateEnv(), episodes=3, seed=10000)
    assert result["unique_initial_states"] == 1
    assert result["success_rate_ci95"] is None
    assert result["mean_episode_length"] == 2.0


def test_warm_start_uses_fresh_stage_steps_and_exploration(tmp_path) -> None:
    source = tmp_path / "parent.pt"
    StateAgent(47, 16, "cpu", StateAgentConfig(hidden_dim=32)).save(source, 600000)
    environments = []
    def factory():
        env = _TinyStateEnv()
        environments.append(env)
        return env
    train_state(StateTrainConfig(
        warm_start=source, total_steps=4, seed_steps=4, hidden_dim=32,
        batch_size=2, replay_capacity=16, eval_every_steps=0, device="cpu",
        output_dir=tmp_path / "new_stage",
    ), env_factory=factory)
    assert len(environments[0].actions) == 4
    assert all(np.max(np.abs(a)) <= 0.1 for a in environments[0].actions)
    assert (tmp_path / "new_stage" / "checkpoint_4.pt").exists()


def test_cli_warm_start_and_resume_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--warm-start", "parent.pt", "--resume", "same.pt"])


def test_resume_preserves_exploration_clock(tmp_path) -> None:
    source = StateAgent(47, 16, "cpu", StateAgentConfig(hidden_dim=32))
    for parameter in source.actor.parameters():
        parameter.data.zero_()
    source.save(tmp_path / "parent.pt", 10)
    environments = []
    def factory():
        env = _TinyStateEnv()
        environments.append(env)
        return env
    train_state(StateTrainConfig(
        resume=tmp_path / "parent.pt", total_steps=14, seed_steps=0,
        behavior_stddev_schedule="linear(0.2,0.0,10)",
        hidden_dim=32, batch_size=16, replay_capacity=16,
        eval_every_steps=0, device="cpu", output_dir=tmp_path / "resumed",
    ), env_factory=factory)
    assert len(environments[0].actions) == 4
    for action in environments[0].actions:
        np.testing.assert_array_equal(action, np.zeros(16))


def test_incompatible_checkpoint_closes_environments(tmp_path) -> None:
    source = StateAgent(64, 16, "cpu", StateAgentConfig(hidden_dim=32))
    source.save(tmp_path / "parent.pt", 10)
    environments = []
    def factory():
        env = _TinyStateEnv()
        environments.append(env)
        return env
    with pytest.raises(ValueError, match="state_dim"):
        train_state(StateTrainConfig(
            warm_start=tmp_path / "parent.pt", hidden_dim=32, device="cpu",
            output_dir=tmp_path / "invalid",
        ), env_factory=factory)
    assert all(env.closed for env in environments)


def test_existing_run_is_not_overwritten(tmp_path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("existing run\n")
    with pytest.raises(FileExistsError, match="output"):
        train_state(StateTrainConfig(output_dir=tmp_path, total_steps=0), env_factory=_TinyStateEnv)
    assert metrics.read_text() == "existing run\n"


def test_zero_action_comparison_is_logged(tmp_path) -> None:
    train_state(StateTrainConfig(
        total_steps=2, seed_steps=2, hidden_dim=32, device="cpu",
        eval_every_steps=2, eval_episodes=1, replay_capacity=16,
        output_dir=tmp_path,
    ), env_factory=_TinyStateEnv)
    events = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    baseline = next(x for x in events if x["type"] == "zero_action_evaluation")
    evaluation = next(x for x in events if x["type"] == "evaluation")
    assert baseline["success_rate"] == 1.0
    assert evaluation["zero_action_success_rate"] == 1.0
    assert evaluation["success_rate_gain_over_zero"] == 0.0

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from contextlib import closing
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .agent import schedule
from .replay import NStepAccumulator, StateReplayBuffer
from .state_agent import StateAgent, StateAgentConfig
from .state_env import OrcaStateCubeEnv
from .train import select_device, write_metric


@dataclass(frozen=True)
class StateTrainConfig:
    preset: str = "single_goal"
    total_steps: int = 1_000_000
    seed_steps: int = 5_000
    seed_action_scale: float = 0.1
    behavior_stddev_schedule: str = "linear(0.2,0.05,100000)"
    behavior_stddev_clip: float = 0.2
    behavior_regularization_alpha: float | None = 0.1
    eval_every_steps: int = 10_000
    eval_episodes: int = 100
    eval_seed: int = 10_000
    checkpoint_every_steps: int = 50_000
    replay_capacity: int = 200_000
    nstep: int = 3
    batch_size: int = 256
    gamma: float = 0.99
    seed: int = 1
    device: str = "auto"
    output_dir: Path | None = None
    resume: Path | None = None
    warm_start: Path | None = None
    max_delta_degrees: float = 3.0
    hidden_dim: int = 1024
    fix_wrist: bool = True
    joint_velocity_limit: float = 10.0
    workspace_radius: float = 0.25
    linear_velocity_limit: float = 2.0
    angular_velocity_limit: float = 20.0
    drop_penalty: float = 10.0
    drop_height: float = 0.10
    success_height: float = 0.12
    max_success_linear_speed: float = 0.15
    max_success_angular_speed: float = 2.0
    progress_reward_scale: float = 5.0
    success_bonus: float = 10.0
    step_penalty: float = 0.01
    gate_orientation_progress_on_grasp: bool = True
    grasp_height_reward_scale: float = 0.01
    stable_hold_reward_scale: float = 0.1
    reset_settle_duration_s: float = 1.0
    cube_pos_xy_jitter: float = 0.001
    target_relative_to_reset: bool = True
    target_policy: str | None = None
    target_sequence_length: int | None = None
    target_rotation_angle_rad: float | None = None
    success_tolerance_rad: float | None = None
    control_period_s: float | None = None
    max_task_duration_s: float | None = None
    success_hold_duration_s: float | None = None
    reward_mode: str | None = None

    def __post_init__(self) -> None:
        alpha = self.behavior_regularization_alpha
        if alpha is not None and (not np.isfinite(alpha) or alpha < 0):
            raise ValueError("behavior_regularization_alpha must be finite and non-negative")
        if self.resume is not None and self.warm_start is not None:
            raise ValueError("resume and warm_start are mutually exclusive")
        for name in ("stable_hold_reward_scale", "reset_settle_duration_s", "cube_pos_xy_jitter"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not (
            np.isfinite(self.seed_action_scale)
            and 0.0 <= self.seed_action_scale <= 1.0
        ):
            raise ValueError("seed_action_scale must be finite and within [0, 1]")
        if not (
            np.isfinite(self.behavior_stddev_clip)
            and self.behavior_stddev_clip >= 0.0
        ):
            raise ValueError(
                "behavior_stddev_clip must be finite and non-negative"
            )
        try:
            schedule(self.behavior_stddev_schedule, step=0)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "behavior_stddev_schedule must be a valid non-negative schedule"
            ) from error


_PRESETS = {
    "turn_30": {
        "target_policy": "fixed_quarter_turn",
        "target_sequence_length": 1,
        "target_rotation_angle_rad": float(np.deg2rad(30.0)),
        "success_tolerance_rad": float(np.deg2rad(15.0)),
    },
    "turn_45": {
        "target_policy": "fixed_quarter_turn",
        "target_sequence_length": 1,
        "target_rotation_angle_rad": float(np.deg2rad(45.0)),
        "success_tolerance_rad": float(np.deg2rad(15.0)),
    },
    "turn_60": {
        "target_policy": "fixed_quarter_turn",
        "target_sequence_length": 1,
        "target_rotation_angle_rad": float(np.deg2rad(60.0)),
        "success_tolerance_rad": float(np.deg2rad(15.0)),
    },
    "single_goal": {
        "target_policy": "fixed_quarter_turn",
        "target_sequence_length": 1,
        "target_rotation_angle_rad": np.pi / 2.0,
        "success_tolerance_rad": 0.4,
    },
    "right_angle": {
        "target_policy": "random_quarter_turn",
        "target_sequence_length": 1,
        "target_rotation_angle_rad": np.pi / 2.0,
        "success_tolerance_rad": 0.4,
    },
    "multi_goal": {
        "target_policy": "cube_orientation_bag",
        "target_sequence_length": 20,
        "target_rotation_angle_rad": np.pi / 2.0,
        "success_tolerance_rad": float(np.deg2rad(15.0)),
    },
}


@dataclass
class _EpisodeFunnel:
    success_tolerance_rad: float
    best_orientation_error_rad: float = float("inf")
    reached_error_60deg: bool = False
    reached_error_45deg: bool = False
    reached_error_30deg: bool = False
    reached_success_tolerance: bool = False
    max_stable_success_steps: int = 0
    drop_step: int | None = None
    success_tolerance_cube_height_m: float | None = None
    success_tolerance_linear_speed: float | None = None
    success_tolerance_angular_speed: float | None = None
    success_tolerance_hand_contact_count: int | None = None

    def observe(self, info: dict, episode_step: int) -> None:
        errors = [info.get("orientation_error_rad")]
        completed_error = info.get("completed_target_orientation_error_rad")
        if completed_error is not None:
            errors.append(completed_error)
        for error in errors:
            if error is None:
                continue
            error = float(error)
            self.best_orientation_error_rad = min(
                self.best_orientation_error_rad, error
            )
            self.reached_error_60deg = bool(
                self.reached_error_60deg or error <= np.deg2rad(60.0)
            )
            self.reached_error_45deg = bool(
                self.reached_error_45deg or error <= np.deg2rad(45.0)
            )
            self.reached_error_30deg = bool(
                self.reached_error_30deg or error <= np.deg2rad(30.0)
            )
            if not self.reached_success_tolerance and (
                error <= self.success_tolerance_rad
            ):
                self.reached_success_tolerance = True
                cube_pos = info.get("cube_pos")
                if cube_pos is not None:
                    self.success_tolerance_cube_height_m = float(cube_pos[2])
                linear_speed = info.get("cube_linear_speed")
                if linear_speed is not None:
                    self.success_tolerance_linear_speed = float(linear_speed)
                angular_speed = info.get("cube_angular_speed")
                if angular_speed is not None:
                    self.success_tolerance_angular_speed = float(angular_speed)
                contact_count = info.get("cube_hand_contact_count")
                if contact_count is not None:
                    self.success_tolerance_hand_contact_count = int(
                        contact_count
                    )
        self.max_stable_success_steps = max(
            self.max_stable_success_steps,
            int(info.get("stable_success_steps", 0)),
        )
        if info.get("target_completed"):
            self.max_stable_success_steps = max(
                self.max_stable_success_steps,
                int(info.get("success_hold_steps", 0)),
            )
        if self.drop_step is None and info.get("dropped"):
            self.drop_step = int(episode_step)

    def metrics(self) -> dict:
        return asdict(self)


def _episode_funnel(reset_info: dict, env) -> _EpisodeFunnel:
    tolerance = reset_info.get("success_tolerance_rad")
    if tolerance is None:
        tolerance = getattr(env, "success_tolerance_rad", 0.4)
    funnel = _EpisodeFunnel(float(tolerance))
    funnel.observe(reset_info, episode_step=0)
    return funnel


def _mean_present(metrics: list[dict], field: str) -> float | None:
    values = [item[field] for item in metrics if item[field] is not None]
    return float(np.mean(values)) if values else None


def _wilson_interval(successes: int, trials: int) -> dict[str, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * np.sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return {
        "lower": float(max(0.0, center - margin)),
        "upper": float(min(1.0, center + margin)),
    }


def resolve_preset(config: StateTrainConfig) -> StateTrainConfig:
    if config.preset not in _PRESETS:
        raise ValueError(f"Unknown state curriculum preset: {config.preset!r}")
    defaults = {
        **_PRESETS[config.preset],
        "control_period_s": 0.08,
        "max_task_duration_s": 8.0,
        "success_hold_duration_s": 2.0,
        "reward_mode": "angular_progress",
    }
    resolved = {
        field: getattr(config, field) if getattr(config, field) is not None else value
        for field, value in defaults.items()
    }
    output_dir = config.output_dir or Path(
        f"runs/state_cube_{config.preset}_seed{config.seed}"
    )
    return replace(config, output_dir=output_dir, **resolved)


def _rotation_bucket(angle_rad: float) -> str | None:
    for label, expected in (
        ("90", np.pi / 2.0),
        ("120", 2.0 * np.pi / 3.0),
        ("180", np.pi),
    ):
        if abs(float(angle_rad) - expected) <= 1e-5:
            return label
    return None


def evaluate_state(agent, env, episodes: int, seed: int, gamma: float = 0.99) -> dict:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and within [0, 1]")
    initial_states = set()
    episode_lengths = []
    returns = []
    successes = []
    drops = []
    timeouts = []
    targets_completed = []
    completion_seconds = []
    final_errors = []
    best_errors = []
    funnel_metrics = []
    bucket_stats: dict[str, dict[str, object]] = {}
    initial_values, discounted_returns, value_biases = [], [], []
    action_squared_sum, saturated_actions, action_count = 0.0, 0, 0
    control_metrics = {name: [] for name in (
        "joint_target_limit_fraction", "joint_position_limit_fraction",
        "controller_tracking_error_rms_deg",
    )}

    def record_attempt(angle_rad: float | None) -> None:
        if angle_rad is None:
            return
        bucket = _rotation_bucket(angle_rad)
        if bucket is not None:
            stats = bucket_stats.setdefault(
                bucket,
                {"attempts": 0, "completed": 0, "completion_seconds": []},
            )
            stats["attempts"] = int(stats["attempts"]) + 1

    for episode in range(episodes):
        observation, reset_info = env.reset(seed=seed + episode)
        initial_states.add(hashlib.sha256(
            np.asarray(observation["state"], dtype=np.float32).tobytes()
        ).digest())
        funnel = _episode_funnel(reset_info, env)
        record_attempt(reset_info.get("target_rotation_angle_rad"))
        episode_return = 0.0
        success = False
        dropped = False
        done = False
        info = reset_info
        episode_steps = 0
        discounted_return = 0.0
        initial_value = agent.value(observation) if hasattr(agent, "value") else None
        if initial_value is not None:
            initial_values.append(initial_value)

        while not done:
            action = agent.act(observation, step=0, eval_mode=True)
            action_squared_sum += float(np.square(action).sum())
            saturated_actions += int(np.sum(np.abs(action) > 0.95))
            action_count += int(np.size(action))
            observation, reward, terminated, truncated, info = env.step(action)
            discounted_return += gamma ** episode_steps * float(reward)
            for name, values in control_metrics.items():
                if name in info:
                    values.append(float(info[name]))
            episode_steps += 1
            funnel.observe(info, episode_step=episode_steps)
            episode_return += float(reward)
            success = success or bool(info.get("is_success", False))
            dropped = dropped or bool(info.get("dropped", False))
            done = terminated or truncated

            if info.get("target_completed"):
                completed_steps = info.get("completed_target_steps")
                if completed_steps is not None:
                    seconds = float(completed_steps) * float(
                        info["control_period_s"]
                    )
                    completion_seconds.append(seconds)
                    bucket = _rotation_bucket(
                        info.get("completed_target_rotation_angle_rad")
                    )
                    if bucket is not None:
                        stats = bucket_stats.setdefault(
                            bucket,
                            {
                                "attempts": 0,
                                "completed": 0,
                                "completion_seconds": [],
                            },
                        )
                        stats["completed"] = int(stats["completed"]) + 1
                        stats["completion_seconds"].append(seconds)
                if not done:
                    record_attempt(info.get("target_rotation_angle_rad"))

        returns.append(episode_return)
        discounted_returns.append(discounted_return)
        if terminated and initial_value is not None:
            value_biases.append(initial_value - discounted_return)
        episode_lengths.append(episode_steps)
        successes.append(success)
        drops.append(dropped)
        timeouts.append(info.get("termination_reason") == "task_timeout")
        targets_completed.append(int(info.get("tasks_completed", int(success))))
        final_errors.append(float(info["orientation_error_rad"]))
        best_errors.append(funnel.best_orientation_error_rad)
        funnel_metrics.append(funnel.metrics())

    rotation_buckets = {}
    for bucket in ("90", "120", "180"):
        if bucket not in bucket_stats:
            continue
        stats = bucket_stats[bucket]
        attempts = int(stats["attempts"])
        completed = int(stats["completed"])
        seconds = stats["completion_seconds"]
        rotation_buckets[bucket] = {
            "attempts": attempts,
            "completed": completed,
            "completion_rate": completed / attempts if attempts else 0.0,
            "mean_completion_s": float(np.mean(seconds)) if seconds else 0.0,
        }

    funnel = {
        "reached_error_60deg_rate": float(
            np.mean([item["reached_error_60deg"] for item in funnel_metrics])
        ),
        "reached_error_45deg_rate": float(
            np.mean([item["reached_error_45deg"] for item in funnel_metrics])
        ),
        "reached_error_30deg_rate": float(
            np.mean([item["reached_error_30deg"] for item in funnel_metrics])
        ),
        "reached_success_tolerance_rate": float(
            np.mean(
                [item["reached_success_tolerance"] for item in funnel_metrics]
            )
        ),
        "mean_max_stable_success_steps": float(
            np.mean(
                [item["max_stable_success_steps"] for item in funnel_metrics]
            )
        ),
        "mean_drop_step": _mean_present(funnel_metrics, "drop_step"),
        "mean_success_tolerance_cube_height_m": _mean_present(
            funnel_metrics, "success_tolerance_cube_height_m"
        ),
        "mean_success_tolerance_linear_speed": _mean_present(
            funnel_metrics, "success_tolerance_linear_speed"
        ),
        "mean_success_tolerance_angular_speed": _mean_present(
            funnel_metrics, "success_tolerance_angular_speed"
        ),
        "mean_success_tolerance_hand_contact_count": _mean_present(
            funnel_metrics, "success_tolerance_hand_contact_count"
        ),
    }

    return {
        "mean_initial_q": float(np.mean(initial_values)) if initial_values else None,
        "mean_discounted_return": float(np.mean(discounted_returns)),
        "mean_value_bias_terminal": float(np.mean(value_biases)) if value_biases else None,
        "value_calibration_episodes": len(value_biases),
        "action_rms": float(np.sqrt(action_squared_sum / action_count)),
        "action_saturation_fraction": saturated_actions / action_count,
        **{f"mean_{name}": float(np.mean(values)) if values else None
           for name, values in control_metrics.items()},
        "episodes": episodes,
        "unique_initial_states": len(initial_states),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "success_rate_ci95": (
            _wilson_interval(int(np.sum(successes)), len(successes))
            if len(initial_states) == episodes else None
        ),
        "mean_targets_completed": float(np.mean(targets_completed)),
        "median_targets_completed": float(np.median(targets_completed)),
        "total_targets_completed": int(np.sum(targets_completed)),
        "drop_rate": float(np.mean(drops)),
        "timeout_rate": float(np.mean(timeouts)),
        "mean_seconds_per_completed_target": (
            float(np.mean(completion_seconds)) if completion_seconds else 0.0
        ),
        "median_seconds_per_completed_target": (
            float(np.median(completion_seconds)) if completion_seconds else 0.0
        ),
        "mean_final_orientation_error_rad": float(np.mean(final_errors)),
        "mean_best_orientation_error_rad": float(np.mean(best_errors)),
        "funnel": funnel,
        "rotation_buckets": rotation_buckets,
    }


def make_state_env(config: StateTrainConfig) -> OrcaStateCubeEnv:
    config = resolve_preset(config)
    return OrcaStateCubeEnv(
        max_delta_degrees=config.max_delta_degrees,
        fixed_joint_names=("right_wrist",) if config.fix_wrist else (),
        joint_velocity_limit=config.joint_velocity_limit,
        workspace_radius=config.workspace_radius,
        linear_velocity_limit=config.linear_velocity_limit,
        angular_velocity_limit=config.angular_velocity_limit,
        goal_mode="cube_orientation",
        target_policy=config.target_policy,
        target_sequence_length=config.target_sequence_length,
        target_rotation_angle_rad=config.target_rotation_angle_rad,
        success_tolerance_rad=config.success_tolerance_rad,
        control_period_s=config.control_period_s,
        max_task_duration_s=config.max_task_duration_s,
        success_hold_duration_s=config.success_hold_duration_s,
        drop_penalty=config.drop_penalty,
        drop_height=config.drop_height,
        success_height=config.success_height,
        max_success_linear_speed=config.max_success_linear_speed,
        max_success_angular_speed=config.max_success_angular_speed,
        reward_mode=config.reward_mode,
        progress_reward_scale=config.progress_reward_scale,
        success_bonus=config.success_bonus,
        step_penalty=config.step_penalty,
        gate_orientation_progress_on_grasp=config.gate_orientation_progress_on_grasp,
        grasp_height_reward_scale=config.grasp_height_reward_scale,
        stable_hold_reward_scale=config.stable_hold_reward_scale,
        reset_settle_duration_s=config.reset_settle_duration_s,
        cube_pos_xy_jitter=config.cube_pos_xy_jitter,
        target_relative_to_reset=config.target_relative_to_reset,
    )


class _ZeroActionAgent:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def act(self, observation, step, eval_mode=False):
        return np.zeros(self.action_dim, dtype=np.float32)


def train_state(
    config: StateTrainConfig,
    env_factory: Callable[[], object] | None = None,
) -> None:
    config = resolve_preset(config)
    if config.output_dir is None:
        raise RuntimeError("Resolved state training output directory is missing")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    metrics_path = config.output_dir / "metrics.jsonl"
    if (metrics_path.exists() or (config.output_dir / "config.json").exists()
            or any(config.output_dir.glob("checkpoint_*.pt"))):
        raise FileExistsError("Training output already exists; choose a fresh output directory")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if env_factory is None:
        env_factory = lambda: make_state_env(config)

    with closing(env_factory()) as env, closing(env_factory()) as eval_env:
        observation, reset_info = env.reset(seed=config.seed)
        state_dim = int(observation["state"].size)
        action_dim = int(env.action_space.shape[0])
        device = select_device(config.device)
        agent = StateAgent(
            state_dim,
            action_dim,
            device,
            StateAgentConfig(
                hidden_dim=config.hidden_dim,
                batch_size=config.batch_size,
                stddev_schedule=config.behavior_stddev_schedule,
                stddev_clip=config.behavior_stddev_clip,
                behavior_regularization_alpha=config.behavior_regularization_alpha,
            ),
        )
        replay = StateReplayBuffer(
            config.replay_capacity, state_dim, action_dim, config.seed
        )
        accumulator = NStepAccumulator(config.nstep, config.gamma)
        start_step = agent.load(config.resume) if config.resume else 0
        if config.warm_start is not None:
            agent.load_actor(config.warm_start)
        episode_return = 0.0
        episode_length = 0
        episode_funnel = _episode_funnel(reset_info, env)

        (config.output_dir / "config.json").write_text(
            json.dumps(asdict(config), default=str, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        zero_baseline = None
        if config.eval_every_steps:
            zero_baseline = evaluate_state(
                _ZeroActionAgent(action_dim), eval_env,
                config.eval_episodes, config.eval_seed,
                gamma=config.gamma,
            )
            event = {"type": "zero_action_evaluation", "step": start_step, **zero_baseline}
            write_metric(metrics_path, event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
        for step in range(start_step + 1, config.total_steps + 1):
            stage_step = step - start_step
            if stage_step <= config.seed_steps:
                action = rng.uniform(
                    -config.seed_action_scale,
                    config.seed_action_scale,
                    size=action_dim,
                ).astype(np.float32)
            else:
                action = agent.act(observation, step)
            next_observation, reward, terminated, truncated, info = env.step(
                action
            )
            done = terminated or truncated
            for transition in accumulator.add(
                observation,
                action,
                reward,
                next_observation,
                terminated,
                episode_end=done,
            ):
                replay.add(
                    transition.observation,
                    transition.action,
                    transition.reward,
                    transition.discount,
                    transition.next_observation,
                )
            observation = next_observation
            episode_return += float(reward)
            episode_length += 1
            episode_funnel.observe(info, episode_step=episode_length)

            if stage_step > config.seed_steps and len(replay) >= config.batch_size:
                update_metrics = agent.update(replay, step)
                if update_metrics:
                    write_metric(
                        metrics_path,
                        {"type": "update", "step": step, **update_metrics},
                    )

            if done:
                write_metric(
                    metrics_path,
                    {
                        "type": "train_episode",
                        "step": step,
                        "return": episode_return,
                        "length": episode_length,
                        "success": bool(info.get("is_success", False)),
                        "dropped": bool(info.get("dropped", False)),
                        "tasks_completed": int(info.get("tasks_completed", 0)),
                        "termination_reason": info.get("termination_reason"),
                        "final_orientation_error_rad": info.get(
                            "orientation_error_rad"
                        ),
                        **episode_funnel.metrics(),
                    },
                )
                observation, reset_info = env.reset()
                episode_return = 0.0
                episode_length = 0
                episode_funnel = _episode_funnel(reset_info, env)

            if config.eval_every_steps and step % config.eval_every_steps == 0:
                result = evaluate_state(
                    agent, eval_env, config.eval_episodes, config.eval_seed, gamma=config.gamma
                )
                event = {"type": "evaluation", "step": step, **result}
                event["zero_action_success_rate"] = zero_baseline["success_rate"]
                event["success_rate_gain_over_zero"] = (
                    result["success_rate"] - zero_baseline["success_rate"]
                )
                write_metric(metrics_path, event)
                print(json.dumps(event, ensure_ascii=False), flush=True)

            if (
                config.checkpoint_every_steps
                and step % config.checkpoint_every_steps == 0
            ):
                agent.save(config.output_dir / f"checkpoint_{step}.pt", step)

        final_path = config.output_dir / f"checkpoint_{config.total_steps}.pt"
        if not final_path.exists():
            agent.save(final_path, config.total_steps)


def parse_args(argv: list[str] | None = None) -> StateTrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a state-based agent on full cube reorientation"
    )
    parser.add_argument(
        "--preset",
        choices=(
            "turn_30",
            "turn_45",
            "turn_60",
            "single_goal",
            "right_angle",
            "multi_goal",
        ),
        default="single_goal",
    )
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--seed-steps", type=int, default=5_000)
    parser.add_argument("--seed-action-scale", type=float, default=0.1)
    parser.add_argument(
        "--behavior-stddev-schedule",
        default="linear(0.2,0.05,100000)",
    )
    parser.add_argument("--behavior-stddev-clip", type=float, default=0.2)
    regularization = parser.add_mutually_exclusive_group()
    regularization.add_argument("--behavior-regularization-alpha", type=float, default=0.1)
    regularization.add_argument(
        "--no-behavior-regularization", dest="behavior_regularization_alpha",
        action="store_const", const=None,
    )
    parser.add_argument("--eval-every-steps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=50_000)
    parser.add_argument("--replay-capacity", type=int, default=200_000)
    parser.add_argument("--nstep", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument("--output-dir", type=Path)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument("--warm-start", type=Path)
    parser.add_argument("--max-delta-degrees", type=float, default=3.0)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--fix-wrist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--joint-velocity-limit", type=float, default=10.0)
    parser.add_argument("--workspace-radius", type=float, default=0.25)
    parser.add_argument("--linear-velocity-limit", type=float, default=2.0)
    parser.add_argument("--angular-velocity-limit", type=float, default=20.0)
    parser.add_argument("--drop-penalty", type=float, default=10.0)
    parser.add_argument("--drop-height", type=float, default=0.10)
    parser.add_argument("--success-height", type=float, default=0.12)
    parser.add_argument("--max-success-linear-speed", type=float, default=0.15)
    parser.add_argument("--max-success-angular-speed", type=float, default=2.0)
    parser.add_argument("--progress-reward-scale", type=float, default=5.0)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument(
        "--gate-orientation-progress-on-grasp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--grasp-height-reward-scale", type=float, default=0.01
    )
    parser.add_argument("--stable-hold-reward-scale", type=float, default=0.1)
    parser.add_argument("--reset-settle-duration-s", type=float, default=1.0)
    parser.add_argument("--cube-pos-xy-jitter", type=float, default=0.001)
    parser.add_argument(
        "--target-relative-to-reset", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--target-policy",
        choices=(
            "fixed_quarter_turn",
            "random_quarter_turn",
            "cube_orientation_bag",
        ),
    )
    parser.add_argument("--target-sequence-length", type=int)
    target_angle = parser.add_mutually_exclusive_group()
    target_angle.add_argument("--target-rotation-angle-rad", type=float)
    target_angle.add_argument("--target-rotation-degrees", type=float)
    tolerance = parser.add_mutually_exclusive_group()
    tolerance.add_argument("--success-tolerance-rad", type=float)
    tolerance.add_argument("--success-tolerance-degrees", type=float)
    parser.add_argument("--control-period-s", type=float)
    parser.add_argument("--max-task-duration-s", type=float)
    parser.add_argument("--success-hold-duration-s", type=float)
    parser.add_argument(
        "--reward-mode",
        choices=("absolute", "progress", "angular_progress"),
    )
    values = vars(parser.parse_args(argv))
    target_degrees = values.pop("target_rotation_degrees")
    if target_degrees is not None:
        values["target_rotation_angle_rad"] = float(
            np.deg2rad(target_degrees)
        )
    degrees = values.pop("success_tolerance_degrees")
    if degrees is not None:
        values["success_tolerance_rad"] = float(np.deg2rad(degrees))
    return StateTrainConfig(**values)


def main() -> None:
    train_state(parse_args())


if __name__ == "__main__":
    main()

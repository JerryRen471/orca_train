from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .replay import NStepAccumulator, StateReplayBuffer
from .state_agent import StateAgent, StateAgentConfig
from .state_env import OrcaStateCubeEnv
from .train import select_device, write_metric


@dataclass(frozen=True)
class StateTrainConfig:
    preset: str = "single_goal"
    total_steps: int = 1_000_000
    seed_steps: int = 5_000
    eval_every_steps: int = 10_000
    eval_episodes: int = 20
    eval_seed: int = 10_000
    checkpoint_every_steps: int = 50_000
    replay_capacity: int = 200_000
    nstep: int = 3
    batch_size: int = 256
    gamma: float = 0.99
    seed: int = 1
    device: str = "auto"
    output_dir: Path = Path("runs/state_cube_single_goal")
    resume: Path | None = None
    max_delta_degrees: float = 3.0
    hidden_dim: int = 1024
    fix_wrist: bool = False
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
    target_policy: str | None = None
    target_sequence_length: int | None = None
    success_tolerance_rad: float | None = None
    control_period_s: float | None = None
    max_task_duration_s: float | None = None
    success_hold_duration_s: float | None = None
    reward_mode: str | None = None


_PRESETS = {
    "single_goal": {
        "target_policy": "fixed_quarter_turn",
        "target_sequence_length": 1,
        "success_tolerance_rad": 0.4,
    },
    "right_angle": {
        "target_policy": "random_quarter_turn",
        "target_sequence_length": 1,
        "success_tolerance_rad": 0.4,
    },
    "multi_goal": {
        "target_policy": "cube_orientation_bag",
        "target_sequence_length": 20,
        "success_tolerance_rad": float(np.deg2rad(15.0)),
    },
}


def resolve_preset(config: StateTrainConfig) -> StateTrainConfig:
    if config.preset not in _PRESETS:
        raise ValueError(f"Unknown state curriculum preset: {config.preset!r}")
    defaults = {
        **_PRESETS[config.preset],
        "control_period_s": 0.08,
        "max_task_duration_s": 8.0,
        "success_hold_duration_s": 0.16,
        "reward_mode": "angular_progress",
    }
    resolved = {
        field: getattr(config, field) if getattr(config, field) is not None else value
        for field, value in defaults.items()
    }
    return replace(config, **resolved)


def _rotation_bucket(angle_rad: float) -> str | None:
    for label, expected in (
        ("90", np.pi / 2.0),
        ("120", 2.0 * np.pi / 3.0),
        ("180", np.pi),
    ):
        if abs(float(angle_rad) - expected) <= 1e-5:
            return label
    return None


def evaluate_state(agent, env, episodes: int, seed: int) -> dict:
    returns = []
    successes = []
    drops = []
    timeouts = []
    targets_completed = []
    completion_seconds = []
    final_errors = []
    best_errors = []
    bucket_stats: dict[str, dict[str, object]] = {}

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
        record_attempt(reset_info.get("target_rotation_angle_rad"))
        initial_error = float(reset_info["orientation_error_rad"])
        best_error = initial_error
        episode_return = 0.0
        success = False
        dropped = False
        done = False
        info = reset_info

        while not done:
            action = agent.act(observation, step=0, eval_mode=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            current_error = float(info["orientation_error_rad"])
            best_error = min(best_error, current_error)
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
        successes.append(success)
        drops.append(dropped)
        timeouts.append(info.get("termination_reason") == "task_timeout")
        targets_completed.append(int(info.get("tasks_completed", int(success))))
        final_errors.append(float(info["orientation_error_rad"]))
        best_errors.append(best_error)

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

    return {
        "return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
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
        "rotation_buckets": rotation_buckets,
    }


def train_state(
    config: StateTrainConfig,
    env_factory: Callable[[], object] | None = None,
) -> None:
    config = resolve_preset(config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "metrics.jsonl"
    (config.output_dir / "config.json").write_text(
        json.dumps(asdict(config), default=str, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if env_factory is None:
        fixed_joint_names = ("right_wrist",) if config.fix_wrist else ()
        task_values = {
            "target_policy": config.target_policy,
            "target_sequence_length": config.target_sequence_length,
            "success_tolerance_rad": config.success_tolerance_rad,
            "control_period_s": config.control_period_s,
            "max_task_duration_s": config.max_task_duration_s,
            "success_hold_duration_s": config.success_hold_duration_s,
            "reward_mode": config.reward_mode,
        }
        if any(value is None for value in task_values.values()):
            raise ValueError("State curriculum must be fully resolved")

        def env_factory() -> OrcaStateCubeEnv:
            return OrcaStateCubeEnv(
                max_delta_degrees=config.max_delta_degrees,
                fixed_joint_names=fixed_joint_names,
                joint_velocity_limit=config.joint_velocity_limit,
                workspace_radius=config.workspace_radius,
                linear_velocity_limit=config.linear_velocity_limit,
                angular_velocity_limit=config.angular_velocity_limit,
                goal_mode="cube_orientation",
                target_policy=config.target_policy,
                target_sequence_length=config.target_sequence_length,
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
            )

    env = env_factory()
    eval_env = env_factory()
    observation, _ = env.reset(seed=config.seed)
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
        ),
    )
    replay = StateReplayBuffer(
        config.replay_capacity, state_dim, action_dim, config.seed
    )
    accumulator = NStepAccumulator(config.nstep, config.gamma)
    start_step = agent.load(config.resume) if config.resume else 0
    episode_return = 0.0
    episode_length = 0

    try:
        for step in range(start_step + 1, config.total_steps + 1):
            if step <= config.seed_steps:
                action = rng.uniform(-1.0, 1.0, size=action_dim).astype(
                    np.float32
                )
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

            if step > config.seed_steps and len(replay) >= config.batch_size:
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
                    },
                )
                observation, _ = env.reset()
                episode_return = 0.0
                episode_length = 0

            if config.eval_every_steps and step % config.eval_every_steps == 0:
                result = evaluate_state(
                    agent, eval_env, config.eval_episodes, config.eval_seed
                )
                event = {"type": "evaluation", "step": step, **result}
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
    finally:
        env.close()
        eval_env.close()


def parse_args(argv: list[str] | None = None) -> StateTrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a state-based agent on full cube reorientation"
    )
    parser.add_argument(
        "--preset",
        choices=("single_goal", "right_angle", "multi_goal"),
        default="single_goal",
    )
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--seed-steps", type=int, default=5_000)
    parser.add_argument("--eval-every-steps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
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
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/state_cube_single_goal")
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-delta-degrees", type=float, default=3.0)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--fix-wrist", action="store_true")
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
        "--target-policy",
        choices=(
            "fixed_quarter_turn",
            "random_quarter_turn",
            "cube_orientation_bag",
        ),
    )
    parser.add_argument("--target-sequence-length", type=int)
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
    degrees = values.pop("success_tolerance_degrees")
    if degrees is not None:
        values["success_tolerance_rad"] = float(np.deg2rad(degrees))
    return StateTrainConfig(**values)


def main() -> None:
    train_state(parse_args())


if __name__ == "__main__":
    main()

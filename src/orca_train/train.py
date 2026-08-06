from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .agent import DrQV2Agent, DrQV2Config
from .env import OrcaVisualCubeEnv
from .replay import ReplayBuffer


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int = 1_000_000
    seed_steps: int = 5_000
    eval_every_steps: int = 10_000
    eval_episodes: int = 5
    checkpoint_every_steps: int = 50_000
    replay_capacity: int = 20_000
    batch_size: int = 256
    gamma: float = 0.99
    seed: int = 1
    device: str = "auto"
    output_dir: Path = Path("runs/drqv2_cube_flip")
    resume: Path | None = None
    max_delta_degrees: float = 3.0
    feature_dim: int = 50
    hidden_dim: int = 1024


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_metric(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


@torch.no_grad()
def evaluate(agent: DrQV2Agent, env, episodes: int, seed: int) -> dict[str, float]:
    returns = []
    successes = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0
        success = False
        while not done:
            action = agent.act(observation, step=0, eval_mode=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            success = success or bool(info.get("is_success", info.get("success", False)))
            done = terminated or truncated
        returns.append(episode_return)
        successes.append(success)
    return {
        "return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
    }


def train(
    config: TrainConfig,
    env_factory: Callable[[], object] | None = None,
) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "metrics.jsonl"
    (config.output_dir / "config.json").write_text(
        json.dumps(asdict(config), default=str, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if env_factory is None:
        env_factory = lambda: OrcaVisualCubeEnv(max_delta_degrees=config.max_delta_degrees)
    env = env_factory()
    eval_env = env_factory()
    observation, _ = env.reset(seed=config.seed)
    device = select_device(config.device)
    agent_config = DrQV2Config(
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        batch_size=config.batch_size,
    )
    agent = DrQV2Agent(observation["pixels"].shape, observation["proprio"].size, 17, device, agent_config)
    replay = ReplayBuffer(config.replay_capacity, config.seed)
    start_step = agent.load(config.resume) if config.resume else 0
    episode_return = 0.0
    episode_length = 0

    try:
        for step in range(start_step + 1, config.total_steps + 1):
            if step <= config.seed_steps:
                action = rng.uniform(-1.0, 1.0, size=17).astype(np.float32)
            else:
                action = agent.act(observation, step)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            discount = 0.0 if terminated else config.gamma
            replay.add(observation, action, reward, discount, next_observation)
            observation = next_observation
            episode_return += reward
            episode_length += 1

            if step > config.seed_steps and len(replay) >= config.batch_size:
                update_metrics = agent.update(replay, step)
                if update_metrics:
                    write_metric(metrics_path, {"type": "update", "step": step, **update_metrics})

            if done:
                write_metric(metrics_path, {
                    "type": "train_episode",
                    "step": step,
                    "return": episode_return,
                    "length": episode_length,
                    "success": bool(info.get("is_success", info.get("success", False))),
                })
                observation, _ = env.reset()
                episode_return = 0.0
                episode_length = 0

            if config.eval_every_steps and step % config.eval_every_steps == 0:
                result = evaluate(agent, eval_env, config.eval_episodes, config.seed + step)
                event = {"type": "evaluation", "step": step, **result}
                write_metric(metrics_path, event)
                print(json.dumps(event, ensure_ascii=False), flush=True)

            if config.checkpoint_every_steps and step % config.checkpoint_every_steps == 0:
                agent.save(config.output_dir / f"checkpoint_{step}.pt", step)

        final_path = config.output_dir / f"checkpoint_{config.total_steps}.pt"
        if not final_path.exists():
            agent.save(final_path, config.total_steps)
    finally:
        env.close()
        eval_env.close()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train DrQ-v2 to flip the Orca cube red face upward")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--seed-steps", type=int, default=5_000)
    parser.add_argument("--eval-every-steps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-every-steps", type=int, default=50_000)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/drqv2_cube_flip"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-delta-degrees", type=float, default=3.0)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

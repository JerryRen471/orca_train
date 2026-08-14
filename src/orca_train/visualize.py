from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agent import DrQV2Agent, DrQV2Config
from .env import OrcaVisualCubeEnv


def infer_action_dim(state: dict) -> int:
    output_weights = state["actor"]["policy.4.weight"]
    return int(output_weights.shape[0])


def run_episode(
    env: OrcaVisualCubeEnv,
    agent: DrQV2Agent,
    viewer: Any,
    *,
    seed: int,
    realtime: bool = True,
) -> dict[str, float | int | bool]:
    observation, _ = env.reset(seed=seed)
    viewer.sync()
    episode_return = 0.0
    success = False
    dropped = False
    length = 0
    frame_period = 1.0 / getattr(env, "metadata", {}).get("render_fps", 30)
    deadline = time.monotonic()

    while viewer.is_running():
        action = agent.act(observation, step=0, eval_mode=True)
        observation, reward, terminated, truncated, info = env.step(action)
        length += 1
        episode_return += reward
        success = success or bool(info.get("is_success", False))
        dropped = dropped or bool(info.get("dropped", False))
        viewer.sync()
        if terminated or truncated:
            break
        if realtime:
            deadline += frame_period
            time.sleep(max(0.0, deadline - time.monotonic()))

    return {
        "seed": seed,
        "return": float(episode_return),
        "length": length,
        "success": success,
        "dropped": dropped,
    }


def load_agent(
    checkpoint: Path,
    observation: dict,
    device: str,
) -> DrQV2Agent:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    config = DrQV2Config(**state["config"])
    action_dim = infer_action_dim(state)
    agent = DrQV2Agent(
        observation["pixels"].shape,
        observation["proprio"].size,
        action_dim,
        device,
        config,
    )
    agent.load(checkpoint)
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained ORCA policy in MuJoCo Viewer")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--random-seeds", action="store_true")
    args = parser.parse_args()

    import mujoco
    import mujoco.viewer

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    action_dim = infer_action_dim(state)
    fixed_joint_names = ("right_wrist",) if action_dim == 16 else ()
    env = OrcaVisualCubeEnv(fixed_joint_names=fixed_joint_names)
    observation, _ = env.reset(seed=args.seed)
    agent = load_agent(args.checkpoint, observation, device)
    print(
        f"Loaded {args.checkpoint} on {device} "
        f"(action_dim={action_dim}, wrist_fixed={action_dim == 16})"
    )
    rng = np.random.default_rng(args.seed)

    try:
        with mujoco.viewer.launch_passive(env.env.model, env.env.data) as viewer:
            camera_id = env.env.model.camera("closeup").id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = camera_id
            for episode in range(args.episodes):
                if not viewer.is_running():
                    break
                episode_seed = (
                    int(rng.integers(0, 2**31))
                    if args.random_seeds
                    else args.seed + episode
                )
                result = run_episode(
                    env,
                    agent,
                    viewer,
                    seed=episode_seed,
                )
                print(f"episode {episode + 1}/{args.episodes}: {result}", flush=True)
                if viewer.is_running() and episode + 1 < args.episodes:
                    time.sleep(max(0.0, args.pause))
    finally:
        env.close()


if __name__ == "__main__":
    main()

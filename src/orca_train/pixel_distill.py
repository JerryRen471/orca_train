from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .agent import RandomShiftsAug
from .guarded_state_policy import GuardedStatePolicy
from .pixel_student import PixelStudent, PixelStudentConfig
from .state_agent import StateAgent, StateAgentConfig
from .train_state import StateTrainConfig, make_state_env


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_task_config(directory):
    values = json.loads((Path(directory) / "config.json").read_text())
    for key in ("output_dir", "resume", "warm_start"):
        if values[key] is not None:
            values[key] = Path(values[key])
    return StateTrainConfig(**values)


def load_teacher(directory):
    directory = Path(directory)
    checkpoint = directory / "checkpoint.pt"
    selection = json.loads((directory / "selection.json").read_text())
    if sha256(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("Teacher checkpoint hash mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    agent = StateAgent(saved["state_dim"], saved["action_dim"], "cpu", StateAgentConfig(**saved["config"]))
    agent.load(checkpoint)
    provenance = {"directory": str(directory.resolve()), "checkpoint_sha256": sha256(checkpoint),
                  "task_config_sha256": sha256(directory / "config.json")}
    controller = directory / "controller.json"
    if controller.exists():
        settings = json.loads(controller.read_text())
        agent = GuardedStatePolicy(agent, **settings)
        provenance["controller"] = settings
        provenance["controller_sha256"] = sha256(controller)
    return agent, provenance


class PixelView:
    """Render the teacher task without changing its resets or accumulated-target actions."""

    def __init__(self, env, config: PixelStudentConfig, renderer=None):
        self.env, self.config = env, config
        self.frames = deque(maxlen=config.frame_stack)
        self.teacher_observation = None
        if renderer is None:
            import mujoco
            env.env.model.cam_fovy[env.env.model.camera(config.camera_name).id] = config.camera_fovy
            renderer = mujoco.Renderer(env.env.model, height=config.image_size, width=config.image_size)
        self.renderer = renderer

    def _frame(self):
        self.renderer.update_scene(self.env.env.data, camera=self.config.camera_name)
        frame = np.asarray(self.renderer.render(), dtype=np.uint8)
        if frame.shape != (self.config.image_size, self.config.image_size, 3):
            raise ValueError("Renderer returned an unexpected image shape")
        return np.ascontiguousarray(frame.transpose(2, 0, 1))

    def observation(self):
        return {"pixels": np.concatenate(tuple(self.frames), axis=0)}

    def reset(self, seed):
        self.teacher_observation, info = self.env.reset(seed=seed)
        frame = self._frame()
        self.frames.clear()
        self.frames.extend(frame.copy() for _ in range(self.config.frame_stack))
        return self.observation(), info

    def step(self, action):
        self.teacher_observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self._frame())
        return self.observation(), reward, terminated, truncated, info

    def close(self):
        self.renderer.close()


@contextmanager
def make_pixel_env(task_config, student_config):
    env = make_state_env(task_config)
    view = None
    try:
        view = PixelView(env, student_config)
        yield view
    finally:
        if view is not None:
            view.close()
        env.close()


def frame_indices(step, frame_stack):
    return np.maximum(np.arange(step - frame_stack + 1, step + 1), 0)


class Demonstrations:
    def __init__(self, directories, config):
        self.config = config
        self.episodes = []
        self.manifests = []
        seen_seeds = set()
        binding = None
        for directory in map(Path, directories):
            manifest = json.loads((directory / "manifest.json").read_text())
            current = (manifest["teacher"], manifest["student_config"])
            if binding is not None and current != binding:
                raise ValueError("Cannot mix different teachers, tasks, or visual contracts")
            binding = current
            if manifest["student_config"] != asdict(config):
                raise ValueError("Dataset visual configuration does not match student")
            self.manifests.append({"path": str((directory / "manifest.json").resolve()),
                                   "sha256": sha256(directory / "manifest.json")})
            for episode in manifest["episodes"]:
                if episode["seed"] in seen_seeds:
                    raise ValueError("Duplicate reset seed in training data")
                seen_seeds.add(episode["seed"])
                path = directory / episode["file"]
                if sha256(path) != episode["sha256"]:
                    raise ValueError("Demonstration hash mismatch")
                with np.load(path, allow_pickle=False) as data:
                    frames, actions = data["frames"].copy(), data["teacher_actions"].copy()
                if (frames.dtype != np.uint8 or frames.shape != (len(actions) + 1, 3, config.image_size, config.image_size)
                        or actions.shape != (len(actions), config.action_dim) or len(actions) == 0
                        or not np.all(np.isfinite(actions)) or np.max(np.abs(actions)) > 1):
                    raise ValueError("Invalid demonstration arrays")
                self.episodes.append((frames, actions))
        if not self.episodes:
            raise ValueError("No demonstration episodes")
        self.seeds = seen_seeds
        self.teacher = binding[0]

    def sample(self, rng, batch_size):
        images, actions = [], []
        for index in rng.integers(len(self.episodes), size=batch_size):
            frames, labels = self.episodes[index]
            step = int(rng.integers(len(labels)))
            images.append(frames[frame_indices(step, self.config.frame_stack)].reshape(self.config.pixel_shape))
            actions.append(labels[step])
        return np.stack(images), np.stack(actions)


def episode_summary(seed, infos, rewards, actions, period):
    info = infos[-1]
    success = bool(info["is_success"])
    return {"seed": seed, "success": success, "dropped": bool(info.get("dropped", False)),
            "termination_reason": info.get("termination_reason"), "steps": len(infos),
            "completion_s": len(infos) * period if success else None,
            "final_error_deg": float(np.rad2deg(info["orientation_error_rad"])),
            "best_error_deg": min(float(np.rad2deg(i["orientation_error_rad"])) for i in infos),
            "return": float(sum(rewards)),
            "joint_target_limit_fraction": float(np.mean([i["joint_target_limit_fraction"] for i in infos])),
            "joint_position_limit_fraction": float(np.mean([i["joint_position_limit_fraction"] for i in infos])),
            "action_rms": float(np.sqrt(np.mean(np.square(actions))))}


def summarize(episodes):
    times = [e["completion_s"] for e in episodes if e["success"]]
    return {"episodes": len(episodes), "success_rate": float(np.mean([e["success"] for e in episodes])),
            "drop_rate": float(np.mean([e["dropped"] for e in episodes])),
            "mean_completion_s": float(np.mean(times)) if times else None,
            **{f"mean_{key}": float(np.mean([e[key] for e in episodes])) for key in [
                "final_error_deg", "best_error_deg", "return", "joint_target_limit_fraction",
                "joint_position_limit_fraction", "action_rms"]}}


def collect(args):
    config = PixelStudentConfig()
    student = None
    student_metadata = None
    if args.student:
        student, student_metadata = PixelStudent.load(args.student)
        config = student.config
    teacher, provenance = load_teacher(args.teacher)
    if student_metadata and student_metadata.get("teacher") != provenance:
        raise ValueError("DAgger teacher does not match the student's training teacher")
    task_config = read_task_config(args.teacher)
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    episodes = []
    with make_pixel_env(task_config, config) as env:
        for offset in range(args.episodes):
            seed = args.seed + offset
            observation, _ = env.reset(seed)
            frames = [env.frames[-1].copy()]
            labels, executed, infos, rewards = [], [], [], []
            for _ in range(round(task_config.max_task_duration_s / task_config.control_period_s)):
                label = teacher.act(env.teacher_observation, step=0, eval_mode=True)
                action = label.copy()
                if student is not None and rng.random() >= args.beta:
                    action = student.act(observation)
                if args.noise:
                    action = np.clip(action + rng.normal(0, args.noise, action.shape), -1, 1).astype(np.float32)
                observation, reward, term, trunc, info = env.step(action)
                frames.append(env.frames[-1].copy())
                labels.append(label.copy()); executed.append(action.copy())
                infos.append(info); rewards.append(reward)
                if term or trunc:
                    break
            path = args.output / f"episode_{seed}.npz"
            np.savez_compressed(path, frames=np.stack(frames), teacher_actions=np.stack(labels),
                                executed_actions=np.stack(executed))
            result = episode_summary(seed, infos, rewards, executed, task_config.control_period_s)
            result.update(file=path.name, sha256=sha256(path))
            episodes.append(result)
            print(json.dumps({"type": "collection", "index": offset + 1, **result}), flush=True)
    manifest = {"format": "orca-pixel-demonstrations-v1", "teacher": provenance,
                "student_config": asdict(config), "student_checkpoint": str(args.student) if args.student else None,
                "student_sha256": sha256(args.student) if args.student else None,
                "teacher_probability": args.beta, "action_noise_std": args.noise,
                "student_inputs": ["pixels"], "label": "teacher action at the exact pre-action state",
                "episodes": episodes, "summary": summarize(episodes)}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"type": "collection_complete", **manifest["summary"]}), flush=True)


def fit(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.resume:
        model, parent = PixelStudent.load(args.resume, args.device)
    else:
        model, parent = PixelStudent().to(args.device), None
    data = Demonstrations(args.data, model.config)
    if parent is not None and parent["teacher"] != data.teacher:
        raise ValueError("Resume checkpoint and demonstrations use different teachers")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    augmentation = RandomShiftsAug(args.augmentation_pad)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"seed": args.seed, "training_steps": args.steps, "student_inputs": ["pixels"],
                "learning_rate": args.learning_rate, "batch_size": args.batch_size,
                "augmentation_pad": args.augmentation_pad,
                "data": data.manifests, "parent_sha256": sha256(args.resume) if args.resume else None,
                "parent": parent, "optimizer": "fresh Adam", "loss": "mean squared teacher action error"}
    metadata["teacher"] = data.teacher
    metadata["training_reset_seeds"] = sorted(data.seeds)
    metadata["source_sha256"] = {name: sha256(Path(__file__).with_name(name))
                                  for name in ["pixel_distill.py", "pixel_student.py", "agent.py"]}
    write_json(args.output / "training.json", metadata)
    started = time.monotonic()
    losses = []
    model.train()
    for step in range(1, args.steps + 1):
        images, actions = data.sample(rng, args.batch_size)
        images = torch.as_tensor(images, device=args.device).float()
        if args.augmentation_pad:
            images = augmentation(images)
        labels = torch.as_tensor(actions, device=args.device)
        predicted = model(images)
        loss = F.mse_loss(predicted, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite student loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        losses.append(float(loss.item()))
        if step % 100 == 0 or step == args.steps:
            event = {"type": "fit", "step": step, "loss": float(np.mean(losses[-100:])),
                     "elapsed_s": time.monotonic() - started}
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
        if step % args.checkpoint_every == 0 or step == args.steps:
            model.save(args.output / f"checkpoint_{step}.pt", metadata={**metadata, "step": step})
    write_json(args.output / "completion.json", {"steps": args.steps, "elapsed_s": time.monotonic() - started,
                                                "final_loss": float(np.mean(losses[-100:]))})


def evaluate(args):
    task_config = read_task_config(args.teacher)
    if args.teacher_policy:
        policy, _ = load_teacher(args.teacher)
        config = PixelStudentConfig()
    else:
        policy, metadata = PixelStudent.load(args.student)
        config = policy.config
        if metadata["teacher"]["task_config_sha256"] != sha256(args.teacher / "config.json"):
            raise ValueError("Evaluation task does not match the student's training task")
        if set(range(args.seed, args.seed + args.episodes)).intersection(metadata["training_reset_seeds"]):
            raise ValueError("Evaluation seeds overlap demonstration collection")
    episodes = []
    with make_pixel_env(task_config, config) as env:
        for offset in range(args.episodes):
            seed = args.seed + offset
            observation, initial = env.reset(seed)
            actions, rewards, infos = [], [], []
            trajectory = []
            for step in range(round(task_config.max_task_duration_s / task_config.control_period_s)):
                action = (policy.act(env.teacher_observation, step=0, eval_mode=True) if args.teacher_policy
                          else policy.act(observation))
                observation, reward, term, trunc, info = env.step(action)
                actions.append(action); rewards.append(reward); infos.append(info)
                if offset < 3:
                    trajectory.append({"action": action.tolist(), "cube_pos": np.asarray(info["cube_pos"]).tolist(),
                                       "error_deg": float(np.rad2deg(info["orientation_error_rad"]))})
                if term or trunc:
                    break
            result = episode_summary(seed, infos, rewards, actions, task_config.control_period_s)
            result["initial_cube_pos"] = np.asarray(initial["cube_pos"]).tolist()
            if offset < 3:
                result["trajectory"] = trajectory
            episodes.append(result)
            print(json.dumps({"type": "evaluate", "index": offset + 1,
                              **{k: v for k, v in result.items() if k != "trajectory"}}), flush=True)
    report = {"student": str(args.student) if args.student else None,
              "student_sha256": sha256(args.student) if args.student else None,
              "teacher_directory": str(args.teacher), "teacher_policy": args.teacher_policy,
              "task_config_sha256": sha256(args.teacher / "config.json"),
              "student_config": asdict(config), "seed_start": args.seed,
              "student_inputs": ["pixels"] if not args.teacher_policy else ["state"],
              "summary": summarize(episodes), "episodes": episodes}
    write_json(args.output, report)
    print(json.dumps({"type": "evaluation_complete", **report["summary"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Image-only teacher distillation and closed-loop evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["collect", "evaluate"]:
        child = subparsers.add_parser(name)
        child.add_argument("--teacher", type=Path, required=True)
        child.add_argument("--student", type=Path)
        child.add_argument("--episodes", type=int, default=64)
        child.add_argument("--seed", type=int, required=True)
        child.add_argument("--output", type=Path, required=True)
        if name == "collect":
            child.add_argument("--beta", type=float, default=1.0)
            child.add_argument("--noise", type=float, default=0.0)
        else:
            child.add_argument("--teacher-policy", action="store_true")
    child = subparsers.add_parser("fit")
    child.add_argument("--data", nargs="+", type=Path, required=True)
    child.add_argument("--output", type=Path, required=True)
    child.add_argument("--resume", type=Path)
    child.add_argument("--seed", type=int, required=True)
    child.add_argument("--steps", type=int, default=1500)
    child.add_argument("--batch-size", type=int, default=64)
    child.add_argument("--learning-rate", type=float, default=3e-4)
    child.add_argument("--augmentation-pad", type=int, default=2)
    child.add_argument("--checkpoint-every", type=int, default=500)
    child.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    args = parser.parse_args()
    if args.command == "evaluate" and not (args.student or args.teacher_policy):
        parser.error("evaluate requires --student or --teacher-policy")
    if args.command == "collect" and (not 0 <= args.beta <= 1 or not np.isfinite(args.noise) or args.noise < 0):
        parser.error("beta must be in [0, 1] and noise finite and non-negative")
    if args.command == "collect" and not args.student and args.beta != 1:
        parser.error("A mixing probability below one requires --student")
    if args.command == "fit" and (args.augmentation_pad < 0 or not np.isfinite(args.learning_rate)
                                  or args.learning_rate <= 0):
        parser.error("augmentation pad must be non-negative and learning rate finite and positive")
    for key in ["episodes", "steps", "batch_size", "checkpoint_every"]:
        if hasattr(args, key) and getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    torch.set_num_threads(1)
    {"collect": collect, "fit": fit, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()

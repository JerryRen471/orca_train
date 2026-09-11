from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from orca_train.pixel_distill import Demonstrations, PixelView, collect, frame_indices, sha256
from orca_train.pixel_student import PixelStudent, PixelStudentConfig


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_config():
    return PixelStudentConfig(image_size=32, feature_dim=8, hidden_dim=16, action_dim=2)


def test_student_is_image_only_and_export_reloads_without_teacher(tmp_path):
    torch.manual_seed(4)
    model = PixelStudent(small_config()).eval()
    obs = {"pixels": np.zeros(model.config.pixel_shape, np.uint8)}
    action = model.act(obs, step=0)
    assert action.shape == (2,) and np.all(np.abs(action) <= 1)
    np.testing.assert_array_equal(action, model.act(obs, step=9999))
    with pytest.raises(ValueError, match="only"):
        model.act({**obs, "state": np.zeros(64)})
    with pytest.raises(ValueError, match="only"):
        model.act({**obs, "proprio": np.zeros(20)})
    path = tmp_path / "student.pt"
    model.save(path, metadata={"seed": 4})
    saved = torch.load(path, weights_only=True)
    assert set(saved) == {"format", "config", "policy", "metadata"}
    restored, metadata = PixelStudent.load(path)
    np.testing.assert_array_equal(action, restored.act(obs))
    assert metadata == {"seed": 4}
    bright = model.act({"pixels": np.full(model.config.pixel_shape, 255, np.uint8)})
    assert not np.allclose(action, bright)


@pytest.mark.parametrize("pixels", [np.zeros((9, 32, 32), np.float32), np.zeros((3, 32, 32), np.uint8)])
def test_student_rejects_wrong_image_contract(pixels):
    with pytest.raises(ValueError, match="uint8"):
        PixelStudent(small_config()).act({"pixels": pixels})


class FakeTask:
    def __init__(self):
        self.env = SimpleNamespace(data=SimpleNamespace(value=0))
        self.calls = []

    def reset(self, seed):
        self.env.data.value = 10
        self.calls.append(("reset", seed))
        return {"state": np.array([10.])}, {"private": "teacher"}

    def step(self, action):
        self.calls.append(("step", action.copy()))
        self.env.data.value += 1
        return {"state": np.array([self.env.data.value])}, 3., False, False, {"private": "teacher"}


class FakeRenderer:
    def update_scene(self, data, camera):
        self.data = data

    def render(self):
        return np.full((32, 32, 3), self.data.value, np.uint8)

    def close(self):
        pass


def test_pixel_view_preserves_actions_and_never_stacks_across_resets():
    task = FakeTask()
    view = PixelView(task, small_config(), renderer=FakeRenderer())
    obs, _ = view.reset(123)
    assert set(obs) == {"pixels"} and np.all(obs["pixels"] == 10)
    action = np.array([.25, -.4], np.float32)
    obs, reward, _, _, _ = view.step(action)
    np.testing.assert_array_equal(task.calls[-1][1], action)
    assert reward == 3
    np.testing.assert_array_equal(obs["pixels"][:, 0, 0], [10]*6 + [11]*3)
    np.testing.assert_array_equal(view.teacher_observation["state"], [11])
    obs, _ = view.reset(124)
    assert np.all(obs["pixels"] == 10)


def make_dataset(path, seed=1, teacher="abc"):
    path.mkdir()
    config = small_config()
    frames = np.stack([np.full((3, 32, 32), i, np.uint8) for i in [10, 20, 30]])
    episode = path / "episode.npz"
    np.savez_compressed(episode, frames=frames, teacher_actions=np.array([[.1, .2], [.3, .4]], np.float32))
    manifest = {"teacher": {"checkpoint_sha256": teacher}, "student_config": asdict(config),
                "episodes": [{"seed": seed, "file": episode.name, "sha256": sha256(episode)}]}
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_dataset_stack_uses_pre_action_frame_and_never_future(tmp_path):
    path = tmp_path / "data"
    make_dataset(path)
    data = Demonstrations([path], small_config())
    images, labels = data.sample(np.random.default_rng(1), 30)
    for image, label in zip(images, labels):
        if np.isclose(label[0], .1):
            assert np.all(image == 10)
        else:
            np.testing.assert_array_equal(image[:, 0, 0], [10]*6 + [20]*3)
    np.testing.assert_array_equal(frame_indices(0, 3), [0, 0, 0])


def test_dataset_rejects_mixed_teacher_or_repeated_seed(tmp_path):
    first, second, third = [tmp_path / name for name in ["a", "b", "c"]]
    make_dataset(first)
    make_dataset(second, seed=2, teacher="different")
    make_dataset(third)
    with pytest.raises(ValueError, match="different teachers"):
        Demonstrations([first, second], small_config())
    with pytest.raises(ValueError, match="Duplicate reset"):
        Demonstrations([first, third], small_config())


def test_dataset_checks_integrity(tmp_path):
    path = tmp_path / "data"
    make_dataset(path)
    with (path / "episode.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        Demonstrations([path], small_config())


def test_fine_tuning_retains_parent_training_seeds(tmp_path):
    from orca_train.pixel_distill import fit

    directory = tmp_path / 'data'
    make_dataset(directory, seed=1)
    parent = tmp_path / 'parent.pt'
    PixelStudent(small_config()).save(parent, metadata={
        'teacher': {'checkpoint_sha256': 'abc'}, 'training_reset_seeds': [99]})
    fit(SimpleNamespace(seed=2, resume=parent, device='cpu', data=[directory],
                        learning_rate=.0001, augmentation_pad=0, output=tmp_path / 'fit',
                        batch_size=2, steps=1, checkpoint_every=1))
    _, metadata = PixelStudent.load(tmp_path / 'fit/checkpoint_1.pt')
    assert metadata['training_reset_seeds'] == [1, 99]
    assert metadata['parent_sha256'] == sha256(parent)


@pytest.mark.parametrize("wrong_task", [False, True])
def test_evaluation_rejects_training_seeds_or_different_task(tmp_path, monkeypatch, wrong_task):
    import orca_train.pixel_distill as module
    (tmp_path / "config.json").write_text('{}')
    metadata = {"teacher": {"task_config_sha256": 'wrong' if wrong_task else sha256(tmp_path / "config.json")},
                "training_reset_seeds": [7]}
    monkeypatch.setattr(module, "read_task_config", lambda p: None)
    monkeypatch.setattr(PixelStudent, "load", lambda p: (SimpleNamespace(config=small_config()), metadata))
    args = SimpleNamespace(teacher=tmp_path, teacher_policy=False, student=tmp_path / 'student.pt', seed=7, episodes=1)
    with pytest.raises(ValueError, match="task does not match" if wrong_task else "overlap"):
        module.evaluate(args)


def test_teacher_labels_are_pre_action_and_distinct_from_executed_student_actions(tmp_path, monkeypatch):
    import orca_train.pixel_distill as module
    from contextlib import contextmanager

    class Teacher:
        def act(self, obs, **kwargs):
            return np.repeat(obs["state"][0] / 100, 2).astype(np.float32)

    class Student:
        config = small_config()
        def act(self, obs):
            assert set(obs) == {"pixels"}
            return np.array([-.25, -.5], np.float32)

    class Task(FakeTask):
        def step(self, action):
            obs, reward, term, trunc, _ = super().step(action)
            return obs, reward, term, self.env.data.value == 12, {
                "is_success": False, "orientation_error_rad": 1.,
                "joint_target_limit_fraction": 0., "joint_position_limit_fraction": 0.}

    @contextmanager
    def factory(*args):
        yield PixelView(Task(), small_config(), renderer=FakeRenderer())

    student_path = tmp_path / "student.pt"
    student_path.write_bytes(b"student")
    monkeypatch.setattr(module, "make_pixel_env", factory)
    monkeypatch.setattr(module, "load_teacher", lambda p: (Teacher(), {}))
    monkeypatch.setattr(module, "read_task_config", lambda p: SimpleNamespace(max_task_duration_s=2, control_period_s=1))
    monkeypatch.setattr(PixelStudent, "load", lambda p: (Student(), {}))
    output = tmp_path / "data"
    collect(SimpleNamespace(student=student_path, teacher=tmp_path, output=output,
                            seed=123, episodes=1, beta=0., noise=0.))
    with np.load(output / "episode_123.npz") as data:
        np.testing.assert_allclose(data["teacher_actions"], [[.1, .1], [.11, .11]])
        np.testing.assert_allclose(data["executed_actions"], [[-.25, -.5], [-.25, -.5]])
        np.testing.assert_array_equal(data["frames"][:, 0, 0, 0], [10, 11, 12])

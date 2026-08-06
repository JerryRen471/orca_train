import numpy as np
import torch

from orca_train.agent import DrQV2Agent, DrQV2Config, RandomShiftsAug
from orca_train.replay import ReplayBuffer


def make_observation(value: int = 0) -> dict[str, np.ndarray]:
    return {
        "pixels": np.full((9, 84, 84), value, dtype=np.uint8),
        "proprio": np.zeros(17, dtype=np.float32),
    }


def test_random_shift_preserves_shape_and_constant_images():
    augmentation = RandomShiftsAug(pad=4)
    images = torch.full((4, 9, 84, 84), 0.25)

    shifted = augmentation(images)

    assert shifted.shape == images.shape
    torch.testing.assert_close(shifted, images)


def test_replay_buffer_keeps_pixels_compact_and_samples_tensors():
    replay = ReplayBuffer(capacity=8, seed=7)
    observation = make_observation(10)
    next_observation = make_observation(20)
    replay.add(
        observation,
        np.zeros(17, dtype=np.float32),
        reward=1.5,
        discount=0.99,
        next_observation=next_observation,
    )

    batch = replay.sample(1, device=torch.device("cpu"))

    assert replay.pixels.dtype == np.uint8
    assert batch.pixels.dtype == torch.uint8
    assert batch.pixels.shape == (1, 9, 84, 84)
    assert batch.proprio.shape == (1, 17)
    assert batch.actions.shape == (1, 17)


def test_agent_action_and_single_update():
    config = DrQV2Config(
        feature_dim=16,
        hidden_dim=32,
        batch_size=2,
        update_every_steps=1,
    )
    agent = DrQV2Agent(
        pixel_shape=(9, 84, 84),
        proprio_dim=17,
        action_dim=17,
        device="cpu",
        config=config,
    )
    observation = make_observation(30)
    action = agent.act(observation, step=0, eval_mode=True)

    assert action.shape == (17,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    replay = ReplayBuffer(capacity=8, seed=3)
    for index in range(4):
        replay.add(
            make_observation(index),
            np.zeros(17, dtype=np.float32),
            reward=0.25,
            discount=0.99,
            next_observation=make_observation(index + 1),
        )

    metrics = agent.update(replay, step=1)

    assert set(metrics) >= {"critic_loss", "actor_loss", "q"}
    assert all(np.isfinite(value) for value in metrics.values())

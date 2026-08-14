import numpy as np
import pytest
import torch

from orca_train.agent import DrQV2Agent, DrQV2Config, RandomShiftsAug
from orca_train.agent import actor_loss_from_q_values, sample_noisy_action
from orca_train.replay import NStepAccumulator, ReplayBuffer


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

    assert set(metrics) >= {"critic_loss", "actor_loss", "q", "target_q"}
    assert all(np.isfinite(value) for value in metrics.values())


def test_three_step_accumulator_computes_discounted_returns_and_flushes_episode():
    accumulator = NStepAccumulator(nstep=3, gamma=0.9)
    observations = [make_observation(value) for value in range(4)]

    assert accumulator.add(observations[0], np.zeros(17), 1.0, observations[1], False) == []
    assert accumulator.add(observations[1], np.ones(17), 2.0, observations[2], False) == []
    transitions = accumulator.add(
        observations[2], np.full(17, 2.0), 3.0, observations[3], True
    )

    assert len(transitions) == 3
    assert transitions[0].reward == pytest.approx(1.0 + 0.9 * 2.0 + 0.9**2 * 3.0)
    assert transitions[0].discount == 0.0
    assert transitions[1].reward == pytest.approx(2.0 + 0.9 * 3.0)
    assert transitions[2].reward == pytest.approx(3.0)
    assert all(transition.discount == 0.0 for transition in transitions)


def test_actor_uses_reparameterized_noise_and_minimum_twin_q():
    mean = torch.zeros(4, 3, requires_grad=True)
    generator = torch.Generator().manual_seed(7)
    action = sample_noisy_action(mean, stddev=0.2, clip=0.3, generator=generator)
    assert not torch.equal(action, mean)
    assert action.requires_grad

    q1 = torch.tensor([[3.0], [1.0]])
    q2 = torch.tensor([[2.0], [4.0]])
    assert actor_loss_from_q_values(q1, q2) == pytest.approx(-1.5)

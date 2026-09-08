import numpy as np
import pytest
import torch

from orca_train.replay import StateReplayBuffer
from orca_train.state_agent import StateAgent, StateAgentConfig


def test_actor_only_transfer_keeps_fresh_critic_and_rejects_old_observation(tmp_path) -> None:
    source = StateAgent(64, 16, "cpu", StateAgentConfig(hidden_dim=32))
    source.save(tmp_path / "parent.pt", 600000)
    target = StateAgent(64, 16, "cpu", StateAgentConfig(hidden_dim=32))
    critic_before = {key: value.clone() for key, value in target.critic.state_dict().items()}
    assert target.load_actor(tmp_path / "parent.pt") == 600000
    for key, value in target.actor.state_dict().items():
        torch.testing.assert_close(value, source.actor.state_dict()[key])
    for key, value in target.critic.state_dict().items():
        torch.testing.assert_close(value, critic_before[key])
    legacy = StateAgent(47, 16, "cpu", StateAgentConfig(hidden_dim=32))
    legacy.save(tmp_path / "legacy.pt", 600000)
    with pytest.raises(ValueError, match="state_dim"):
        target.load_actor(tmp_path / "legacy.pt")


def state_observation(value: float = 0.0) -> dict[str, np.ndarray]:
    return {"state": np.full(47, value, dtype=np.float32)}


def test_state_replay_stores_vectors_and_samples_device_tensors() -> None:
    replay = StateReplayBuffer(capacity=8, state_dim=47, action_dim=16, seed=7)
    replay.add(
        state_observation(0.25),
        np.zeros(16, dtype=np.float32),
        reward=1.5,
        discount=0.99,
        next_observation=state_observation(0.5),
    )

    batch = replay.sample(1, device=torch.device("cpu"))

    assert replay.states.dtype == np.float32
    assert batch.states.shape == (1, 47)
    assert batch.actions.shape == (1, 16)
    assert batch.rewards.shape == (1, 1)
    assert batch.next_states.shape == (1, 47)
    torch.testing.assert_close(batch.states[0], torch.full((47,), 0.25))

    with pytest.raises(ValueError, match="state shape"):
        replay.add(
            {"state": np.zeros(46, dtype=np.float32)},
            np.zeros(16, dtype=np.float32),
            0.0,
            1.0,
            state_observation(),
        )


def test_state_replay_validates_action_shape() -> None:
    replay = StateReplayBuffer(capacity=8, state_dim=47, action_dim=16)

    with pytest.raises(ValueError, match="action shape"):
        replay.add(
            state_observation(),
            np.zeros(15, dtype=np.float32),
            0.0,
            1.0,
            state_observation(),
        )


def test_state_replay_rejects_sampling_more_items_than_available() -> None:
    replay = StateReplayBuffer(capacity=8, state_dim=47, action_dim=16)

    with pytest.raises(ValueError, match="Replay has 0 items, need 1"):
        replay.sample(1, torch.device("cpu"))


def test_state_agent_action_and_single_update() -> None:
    config = StateAgentConfig(
        hidden_dim=32,
        batch_size=2,
        update_every_steps=1,
    )
    agent = StateAgent(
        state_dim=47,
        action_dim=16,
        device="cpu",
        config=config,
    )

    action = agent.act(state_observation(), step=0, eval_mode=True)

    assert action.shape == (16,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    replay = StateReplayBuffer(capacity=8, state_dim=47, action_dim=16, seed=3)
    for index in range(4):
        replay.add(
            state_observation(index / 10.0),
            np.zeros(16, dtype=np.float32),
            reward=0.25,
            discount=0.99,
            next_observation=state_observation((index + 1) / 10.0),
        )

    metrics = agent.update(replay, step=1)

    assert set(metrics) >= {"critic_loss", "actor_loss", "q", "target_q"}
    assert all(np.isfinite(value) for value in metrics.values())


def test_state_agent_action_validates_and_coerces_state() -> None:
    agent = StateAgent(
        47,
        16,
        "cpu",
        StateAgentConfig(hidden_dim=32),
    )

    action = agent.act(
        {"state": np.zeros(47, dtype=np.float64)},
        step=0,
        eval_mode=True,
    )
    assert action.dtype == np.float32

    with pytest.raises(ValueError, match="state shape"):
        agent.act({"state": np.zeros(46)}, step=0, eval_mode=True)
    with pytest.raises(ValueError, match="finite"):
        agent.act(
            {"state": np.full(47, np.nan)},
            step=0,
            eval_mode=True,
        )


def test_state_agent_behavior_noise_uses_configured_clip() -> None:
    config = StateAgentConfig(
        hidden_dim=32,
        stddev_schedule="1.0",
        stddev_clip=0.2,
    )
    agent = StateAgent(47, 16, "cpu", config)
    for parameter in agent.actor.parameters():
        parameter.data.zero_()

    torch.manual_seed(0)
    action = agent.act(state_observation(), step=0)

    assert np.max(np.abs(action)) <= 0.2 + 1e-7


def test_state_agent_defaults_use_survivable_exploration_schedule() -> None:
    config = StateAgentConfig()

    assert config.stddev_schedule == "linear(0.2,0.05,100000)"
    assert config.stddev_clip == pytest.approx(0.2)


@pytest.mark.parametrize("clip", [-0.1, np.inf, np.nan])
def test_state_agent_rejects_invalid_noise_clip(clip) -> None:
    with pytest.raises(ValueError, match="stddev_clip"):
        StateAgentConfig(stddev_clip=clip)


@pytest.mark.parametrize(
    "spec",
    [
        "nan",
        "-0.1",
        "linear(0.2,0.05,0)",
        "linear(0.2,0.05,-1)",
        "linear(nan,0.05,100)",
        "not-a-schedule",
    ],
)
def test_state_agent_rejects_invalid_noise_schedule(spec) -> None:
    with pytest.raises(ValueError, match="stddev_schedule"):
        StateAgentConfig(stddev_schedule=spec)


def test_state_agent_checkpoint_round_trip_includes_dimensions(tmp_path) -> None:
    config = StateAgentConfig(hidden_dim=32, batch_size=2)
    agent = StateAgent(47, 16, "cpu", config)
    checkpoint = tmp_path / "state_agent.pt"

    agent.save(checkpoint, step=7)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert saved["observation_mode"] == "state"
    assert saved["state_dim"] == 47
    assert saved["action_dim"] == 16
    restored = StateAgent(47, 16, "cpu", config)
    assert restored.load(checkpoint) == 7
    for original, loaded in zip(
        agent.actor.parameters(), restored.actor.parameters()
    ):
        torch.testing.assert_close(original, loaded)


@pytest.mark.parametrize(
    ("field", "incompatible_value"),
    [
        ("observation_mode", "pixels"),
        ("state_dim", 46),
        ("action_dim", 15),
    ],
)
def test_state_agent_checkpoint_rejects_incompatible_metadata(
    tmp_path, field, incompatible_value
) -> None:
    config = StateAgentConfig(hidden_dim=32)
    checkpoint = tmp_path / f"bad_{field}.pt"
    StateAgent(47, 16, "cpu", config).save(checkpoint, step=7)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved[field] = incompatible_value
    torch.save(saved, checkpoint)

    with pytest.raises(ValueError, match=field):
        StateAgent(47, 16, "cpu", config).load(checkpoint)


def test_state_agent_checkpoint_rejects_incompatible_hidden_dim(tmp_path) -> None:
    checkpoint = tmp_path / "hidden_dim.pt"
    StateAgent(
        47,
        16,
        "cpu",
        StateAgentConfig(hidden_dim=32),
    ).save(checkpoint, step=7)

    with pytest.raises(ValueError, match="hidden_dim"):
        StateAgent(
            47,
            16,
            "cpu",
            StateAgentConfig(hidden_dim=64),
        ).load(checkpoint)

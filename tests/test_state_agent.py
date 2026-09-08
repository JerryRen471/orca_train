import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from orca_train.replay import StateReplayBuffer
from orca_train.state_agent import StateAgent, StateAgentConfig


def _training_fixture():
    torch.manual_seed(31)
    agent = StateAgent(3, 2, "cpu", StateAgentConfig(
        hidden_dim=16, batch_size=2, update_every_steps=1,
        stddev_schedule="0", stddev_clip=0,
    ))
    replay = StateReplayBuffer(4, 3, 2, seed=5)
    for value in (0.1, 0.5):
        replay.add({"state": np.full(3, value)}, np.zeros(2), 1.0, 0.99,
                   {"state": np.full(3, value + 0.1)})
    return agent, replay


def test_critic_gets_an_update_before_policy_and_targets_move():
    agent, replay = _training_fixture()
    actor_before = copy.deepcopy(agent.actor.state_dict())
    critic_before = copy.deepcopy(agent.critic.state_dict())
    target_before = copy.deepcopy(agent.critic_target.state_dict())
    first = agent.update(replay, step=1)
    assert any(not torch.equal(v, critic_before[k]) for k, v in agent.critic.state_dict().items())
    for k, v in agent.actor.state_dict().items():
        torch.testing.assert_close(v, actor_before[k], rtol=0, atol=0)
    for k, v in agent.critic_target.state_dict().items():
        torch.testing.assert_close(v, target_before[k], rtol=0, atol=0)
    assert "actor_loss" not in first
    second = agent.update(replay, step=2)
    assert "actor_loss" in second
    assert any(not torch.equal(v, actor_before[k]) for k, v in agent.actor.state_dict().items())


def test_online_policy_change_does_not_change_frozen_bootstrap_target():
    agent, replay = _training_fixture()
    other, other_replay = copy.deepcopy((agent, replay))
    with torch.no_grad():
        other.actor.policy[-1].bias.add_(3)
    torch.manual_seed(19)
    baseline = agent.update(replay, step=1)
    torch.manual_seed(19)
    changed = other.update(other_replay, step=1)
    assert changed["target_q"] == pytest.approx(baseline["target_q"], abs=1e-8)


def test_behavior_schedule_does_not_change_target_smoothing():
    agent, replay = _training_fixture()
    other, other_replay = copy.deepcopy((agent, replay))
    other.config = replace(other.config, stddev_schedule="1.0", stddev_clip=0.9)
    torch.manual_seed(19)
    baseline = agent.update(replay, step=1)
    torch.manual_seed(19)
    changed = other.update(other_replay, step=1)
    assert changed["target_q"] == pytest.approx(baseline["target_q"], abs=1e-8)


def test_resume_preserves_delayed_update_phase_and_target_actor(tmp_path):
    agent, replay = _training_fixture()
    agent.update(replay, step=1)
    agent.save(tmp_path / "resume.pt", step=99)
    restored, _ = _training_fixture()
    assert restored.load(tmp_path / "resume.pt") == 99
    torch.manual_seed(41)
    expected = agent.update(copy.deepcopy(replay), step=100)
    torch.manual_seed(41)
    actual = restored.update(copy.deepcopy(replay), step=100)
    assert "actor_loss" in actual
    assert actual == expected
    for network in ("actor", "actor_target", "critic", "critic_target"):
        for original, loaded in zip(getattr(agent, network).parameters(), getattr(restored, network).parameters()):
            torch.testing.assert_close(original, loaded)


def test_legacy_checkpoint_allows_actor_transfer_but_rejects_optimizer_resume(tmp_path):
    source, _ = _training_fixture()
    checkpoint = tmp_path / "legacy.pt"
    source.save(checkpoint, 20_000)
    saved = torch.load(checkpoint, weights_only=True)
    for field in ("algorithm", "actor_target", "critic_updates"):
        saved.pop(field, None)
    torch.save(saved, checkpoint)
    target, _ = _training_fixture()
    with pytest.raises(ValueError, match="algorithm"):
        target.load(checkpoint)
    target.load_actor(checkpoint)
    for actor, target_actor in zip(target.actor.parameters(), target.actor_target.parameters()):
        torch.testing.assert_close(actor, target_actor)


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
        policy_delay=1,
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


def test_value_estimate_uses_conservative_critic():
    agent, _ = _training_fixture()
    for parameter in agent.critic.parameters():
        parameter.data.zero_()
    agent.critic.q1[-1].bias.data.fill_(4)
    agent.critic.q2[-1].bias.data.fill_(2)
    assert agent.value({"state": np.zeros(3, dtype=np.float32)}) == 2.0


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

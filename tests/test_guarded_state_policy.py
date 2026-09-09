import numpy as np
import pytest
import torch

from orca_train.state_agent import StateAgent, StateAgentConfig


def actor():
    agent = StateAgent(22, 2, 'cpu', StateAgentConfig(hidden_dim=8))
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.zero_()
        agent.actor.policy[-1].bias.fill_(float(np.arctanh(.8)))
    return agent


def observation(angle=90, targets=(0, 0, 0)):
    state = np.zeros(22, dtype=np.float32)
    state[15:19] = [np.cos(np.deg2rad(angle / 2)), np.sin(np.deg2rad(angle / 2)), 0, 0]
    state[-3:] = targets
    return {'state': state}


def guarded(agent=None, **kwargs):
    from orca_train.guarded_state_policy import GuardedStatePolicy
    return GuardedStatePolicy(
        agent or actor(), joint_low=np.full(3, -1.), joint_high=np.full(3, 1.),
        active_actuator_indices=np.array([1, 2]), max_delta_radians=.1, **kwargs)


def test_goal_hold_stops_accumulating_controller_targets():
    policy = guarded()
    np.testing.assert_array_equal(policy.act(observation(angle=5), 0, eval_mode=True), [0, 0])
    np.testing.assert_allclose(policy.act(observation(angle=90), 0, eval_mode=True), [.8, .8], atol=1e-6)


def test_projection_uses_observed_targets_and_leaves_rom_margin():
    policy = guarded()
    action = policy.act(observation(targets=(0, .89, .89)), 0, eval_mode=True)
    np.testing.assert_allclose(action, [.1, .1], atol=1e-6)
    np.testing.assert_allclose(.89 + .1 * action, [.9, .9], atol=1e-7)


def test_hold_recovers_an_outside_target_without_exceeding_action_limit():
    policy = guarded()
    action = policy.act(observation(angle=5, targets=(0, 1, -1)), 0, eval_mode=True)
    np.testing.assert_allclose(action, [-1, 1], atol=1e-6)


def test_multistep_recovery_handles_unequal_rom_and_reordered_actions():
    from orca_train.guarded_state_policy import GuardedStatePolicy
    low, high = np.array([-2., 10., -100.]), np.array([4., 30., 0.])
    active = np.array([2, 0])
    policy = GuardedStatePolicy(
        actor(), joint_low=low, joint_high=high,
        active_actuator_indices=active, max_delta_radians=.2)
    targets = np.array([4., 20., -100.])
    for step in range(30):
        obs = observation(angle=5, targets=2 * (targets - low) / (high - low) - 1)
        action = policy.act(obs, step, eval_mode=True)
        assert np.all(np.abs(action) <= 1)
        if step == 0:
            np.testing.assert_array_equal(action, [1, -1])
        targets[active] += .2 * action
    np.testing.assert_allclose(targets, [3.7, 20., -95.], atol=1e-5, rtol=0)


def test_value_evaluates_the_action_executed_by_the_guard():
    agent = actor()
    with torch.no_grad():
        for parameter in agent.critic.parameters():
            parameter.zero_()
        for net, bias in [(agent.critic.q1, -2), (agent.critic.q2, -3)]:
            net[0].weight[0, 22:] = 1
            net[0].bias[0] = 2
            net[2].weight[0, 0] = 1
            net[4].weight[0, 0] = 1
            net[4].bias[0] = bias
    policy = guarded(agent)
    assert policy.value(observation(angle=5)) == pytest.approx(-1)
    assert policy.value(observation(targets=(0, .89, .89))) == pytest.approx(-.8, abs=1e-6)


@pytest.mark.parametrize('margin', [-.01, .5, float('nan')])
def test_invalid_margin_is_rejected(margin):
    with pytest.raises(ValueError, match='margin'):
        guarded(joint_target_margin_fraction=margin)


def test_invalid_state_cannot_bypass_validation_in_hold_mode():
    policy = guarded()
    obs = observation(angle=5)
    obs['state'][0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        policy.act(obs, 0, eval_mode=True)

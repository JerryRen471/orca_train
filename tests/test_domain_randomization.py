from dataclasses import asdict, replace

import mujoco
import numpy as np
import pytest

from orca_train.domain_randomization import DomainConfig, EpisodeDomain


def model():
    return mujoco.MjModel.from_xml_string('''
    <mujoco><asset><material name="black" rgba=".1 .1 .1 1"/></asset>
    <worldbody><light/><camera name="closeup" pos="0 0 1"/>
    <body name="task_cube"><freejoint/><geom type="box" size=".02 .02 .02" mass=".08"/></body>
    <body><joint name="hinge" damping=".1"/><geom size=".01" material="black"/></body>
    </worldbody><actuator><position joint="hinge" kp="2"/></actuator></mujoco>
    ''')


def test_randomization_is_reproducible_non_accumulating_and_physically_consistent():
    m = model()
    d = mujoco.MjData(m)
    domain = EpisodeDomain(m, DomainConfig())
    original = {name: getattr(m, name).copy() for name in domain.ARRAYS}
    first = domain.reset(42, d)
    arrays = {name: getattr(m, name).copy() for name in domain.ARRAYS}
    image = np.full((32, 32, 3), 80, np.uint8)
    first_image = domain.image(image)
    domain.reset(99, d)
    assert domain.reset(42, d) == first
    np.testing.assert_array_equal(domain.image(image), first_image)
    for name, value in arrays.items():
        np.testing.assert_array_equal(getattr(m, name), value)
    np.testing.assert_allclose(m.body_mass[domain.cube], original['body_mass'][domain.cube] * first['mass_scale'])
    np.testing.assert_allclose(m.body_inertia[domain.cube], original['body_inertia'][domain.cube] * first['mass_scale'])
    np.testing.assert_allclose(m.actuator_gainprm[:, 0], -m.actuator_biasprm[:, 1])
    np.testing.assert_array_equal(m.geom_rgba[m.geom_matid >= 0], original['geom_rgba'][m.geom_matid >= 0])
    domain.restore()
    for name, value in original.items():
        np.testing.assert_array_equal(getattr(m, name), value)


def test_visual_changes_do_not_change_physics_draws_and_zero_ranges_are_identity():
    a, b = model(), model()
    base = DomainConfig()
    first = EpisodeDomain(a, base)
    second = EpisodeDomain(b, replace(base, camera_position_m=.02, noise_std_max=8))
    p = first.reset(123, mujoco.MjData(a))
    q = second.reset(123, mujoco.MjData(b))
    for key in ['mass_scale', 'gain_scale', 'friction_scale', 'damping_scale']:
        assert p[key] == q[key]
    zero = EpisodeDomain(model(), DomainConfig(**{k: 0. for k in asdict(base)}))
    zero.reset(1, mujoco.MjData(zero.model))
    for name, value in zero.nominal.items():
        np.testing.assert_array_equal(getattr(zero.model, name), value)
    image = np.random.default_rng(1).integers(0, 256, (32, 32, 3), np.uint8)
    np.testing.assert_array_equal(zero.image(image), image)


@pytest.mark.parametrize('values', [{'mass_fraction': 1}, {'noise_std_max': -1}, {'gain_fraction': float('nan')}])
def test_invalid_domains_are_rejected(values):
    with pytest.raises(ValueError):
        DomainConfig(**values)


def test_visual_randomization_preserves_task_reset_rng_and_action_dynamics():
    from orca_train.train_state import StateTrainConfig, make_state_env

    a, b = make_state_env(StateTrainConfig()), make_state_env(StateTrainConfig())
    try:
        domain = EpisodeDomain(b.env.model, DomainConfig(mass_fraction=0, friction_fraction=0,
                                                        gain_fraction=0, damping_fraction=0))
        for seed in [31, 32, 31]:
            domain.reset(seed, b.env.data)
            first, _ = a.reset(seed=seed)
            second, _ = b.reset(seed=seed)
            np.testing.assert_array_equal(first['state'], second['state'])
            action = np.linspace(-.1, .1, a.action_space.shape[0], dtype=np.float32)
            for _ in range(3):
                first, reward_a, term_a, trunc_a, _ = a.step(action)
                second, reward_b, term_b, trunc_b, _ = b.step(action)
                np.testing.assert_array_equal(first['state'], second['state'])
                assert (reward_a, term_a, trunc_a) == (reward_b, term_b, trunc_b)
    finally:
        a.close()
        b.close()

import mujoco
import numpy as np
import pytest

from orca_train.train_state import StateTrainConfig, make_state_env, parse_args


def contact_metrics(env):
    task = env.env
    model = task.model
    thumb, fingers, depths = [], [], []
    for index, contact in enumerate(task.data.contact[:task.data.ncon]):
        first, second = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
        cube_first = task._body_is_in_subtree(first, task._cube_body_id)
        cube_second = task._body_is_in_subtree(second, task._cube_body_id)
        hand = second if cube_first else first if cube_second else -1
        if hand < 0 or not task._body_is_in_subtree(hand, task._hand_mount_body_id):
            continue
        depths.append(max(0.0, -float(contact.dist)))
        joints = []
        while hand:
            start, count = model.body_jntadr[hand], model.body_jntnum[hand]
            joints.extend(model.joint(j).name for j in range(start, start + count))
            hand = int(model.body_parentid[hand])
        force = np.zeros(6)
        mujoco.mj_contactForce(model, task.data, index, force)
        if force[0] <= 0.01:
            continue
        normal = np.asarray(contact.frame[:3]) * (1 if cube_first else -1)
        if any(name.startswith("right_t-") for name in joints):
            thumb.append(normal)
        elif any(name.startswith(("right_i-", "right_m-", "right_r-", "right_p-")) for name in joints):
            fingers.append(normal)
    opposed = any(np.dot(t, f) < -0.5 for t in thumb for f in fingers)
    return opposed, max(depths, default=0.0)


@pytest.mark.parametrize("seed", [10000, 10001, 10002])
def test_default_initial_grasp_preserves_opposed_contacts_for_sixteen_seconds(seed):
    with make_state_env(StateTrainConfig(preset="turn_30", max_task_duration_s=16.0)) as env:
        _, initial = env.reset(seed=seed)
        assert contact_metrics(env)[0], "Reset must finish with thumb-to-finger opposition"
        assert env.env.data.time == pytest.approx(1.5)
        assert initial["orientation_error_rad"] == pytest.approx(np.pi / 6)
        target = env._target.copy()
        fraction = (target[1:] - env._joint_low[1:]) / (env._joint_high[1:] - env._joint_low[1:])
        assert np.all((fraction > 0.05) & (fraction < 0.95))
        initial_position = np.asarray(initial["cube_pos"])
        for _ in range(200):
            _, _, terminated, truncated, info = env.step(np.zeros(16, dtype=np.float32))
            assert not info["dropped"]
            assert info["cube_reliably_held"]
            assert contact_metrics(env)[0]
            assert np.linalg.norm(info["cube_pos"] - initial_position) < 0.003
            np.testing.assert_array_equal(env._target, target)
            if terminated or truncated:
                break
        assert info["termination_reason"] == "task_timeout"


def test_default_grasp_starts_without_cube_hand_penetration():
    with make_state_env(StateTrainConfig(preset="turn_30")) as env:
        env.reset(seed=10000, options={"settle_steps": 0, "control_ramp_steps": 0})
        assert contact_metrics(env)[1] <= 0.0005
        assert not np.any(env.env.data.ctrl)


def test_scene_initial_grasp_preserves_legacy_reset():
    with make_state_env(StateTrainConfig(preset="turn_30", initial_grasp="scene")) as env:
        env.reset(seed=10000)
        assert not contact_metrics(env)[0]
        assert env.env.data.time == pytest.approx(1.0)


def test_initial_grasp_cli_and_validation():
    assert parse_args([]).initial_grasp == "thumb_opposed_v1"
    assert parse_args(["--initial-grasp", "scene"]).initial_grasp == "scene"
    with pytest.raises(ValueError, match="initial_grasp"):
        StateTrainConfig(initial_grasp="unknown")

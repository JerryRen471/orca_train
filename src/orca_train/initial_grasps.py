"""Versioned, physically prepared initial grasps for the state cube task."""
import numpy as np


INITIAL_GRASP_NAMES = ("scene", "thumb_opposed_v1")


def initial_grasp_kwargs(name: str) -> dict:
    if name == "scene":
        return {}
    if name != "thumb_opposed_v1":
        raise ValueError(f"Unknown initial_grasp: {name!r}")
    degrees = {
        "right_t-cmc": -14.0,
        "right_t-abd": 2.0,
        "right_t-mcp": 56.0,
        "right_t-pip": 65.0,
        "right_i-mcp": 36.0,
        "right_i-pip": 30.0,
        "right_m-mcp": 47.0,
        "right_m-pip": 47.0,
        "right_r-mcp": 1.0,
        "right_r-pip": -2.0,
    }
    return {
        "initial_cube_pos": (0.17, -0.015, 0.2015),
        "reset_control_targets_by_joint": {
            joint: float(np.deg2rad(value)) for joint, value in degrees.items()
        },
        "reset_control_ramp_duration_s": 0.5,
    }

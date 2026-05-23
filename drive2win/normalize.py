"""Input/output normalization for the navigation network.

The simulation returns sensor values in their natural units (m/s, radians,
meters, [0,1] friction). Neural networks train and infer better when every
input is in roughly the same range. The functions in this file are the
*single source of truth* for normalization across the whole project.

If you ever change a normalization constant, retrain. Mixing training-time
and deployment-time normalization is the #1 reason "my net trained fine but
won't drive."
"""
from __future__ import annotations
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────
SPD_MAX = 20.0      # speed clip (m/s)
DIST_MAX = 100.0    # checkpoint-distance clip (m)
RAY_MAX = 50.0      # raycast clip (m); matches RAYCAST_MAX_RANGE in SensorSystem.ts

RAW_FEATURE_NAMES = [
    "speed",
    "heading_error",
    "checkpoint_distance",
    "ray_0_front",
    "ray_1_+45",
    "ray_2_+90",
    "ray_3_+135",
    "ray_4_back",
    "ray_5_-135",
    "ray_6_-90",
    "ray_7_-45",
    "ground_friction",
]
FEATURE_NAMES = RAW_FEATURE_NAMES
ENGINEERED_FEATURE_NAMES = [
    "speed",
    "heading_error",
    "checkpoint_distance",
    "ray_0_front",
    "ray_1_+45",
    "ray_2_+90",
    "ray_3_+135",
    "ray_4_back",
    "ray_5_-135",
    "ray_6_-90",
    "ray_7_-45",
    "ground_friction",
    "sin_heading_error",
    "cos_heading_error",
    "abs_heading_error",
    "front_obstacle_pressure",
    "front_left_clearance",
    "front_right_clearance",
    "side_clearance_balance",
    "min_front_clearance",
    "distance_remaining",
    "speed_heading_interaction",
]
ACTION_NAMES = ["throttle", "steering"]
N_RAW_FEATURES = 12
N_FEATURES = len(ENGINEERED_FEATURE_NAMES)
N_ACTIONS = 2


def normalize_states(states_raw: np.ndarray) -> np.ndarray:
    """Map raw sensor readings into roughly [-1, 1].

    Args:
        states_raw: shape (N, 12). Columns in RAW_FEATURE_NAMES order.

    Returns:
        float32 array with engineered features. Raw ranges that are physically
        non-negative stay in [0, 1]; signed steering-relevant signals use
        roughly [-1, 1].
    """
    raw = np.asarray(states_raw, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[None, :]

    base = raw.copy()
    base[:, 0] = np.clip(base[:, 0] / SPD_MAX, -1.0, 1.0)       # speed
    base[:, 1] = np.clip(base[:, 1] / np.pi, -1.0, 1.0)         # heading_error
    base[:, 2] = np.clip(base[:, 2] / DIST_MAX, 0.0, 1.0)       # ckpt distance
    base[:, 3:11] = np.clip(base[:, 3:11] / RAY_MAX, 0.0, 1.0)  # 8 rays
    base[:, 11] = np.clip(base[:, 11], 0.0, 1.2) / 1.2          # friction

    heading = raw[:, 1]
    rays = base[:, 3:11]
    front = rays[:, 0]
    front_left = rays[:, 1]
    front_right = rays[:, 7]
    left = rays[:, 2]
    right = rays[:, 6]
    min_front = np.minimum.reduce([front_left, front, front_right])

    engineered = np.column_stack(
        [
            base,
            np.sin(heading),
            np.cos(heading),
            np.abs(base[:, 1]),
            1.0 - front,
            front_left,
            front_right,
            np.clip((left + front_left - right - front_right) * 0.5, -1.0, 1.0),
            min_front,
            1.0 - base[:, 2],
            base[:, 0] * base[:, 1],
        ]
    )
    return engineered.astype(np.float32)


def sensors_to_input(sensors: dict) -> np.ndarray:
    """Convert a live sensor dict (from `client.get_sensors()` or the WS
    `state['sensors']`) to the normalized 12-vector the network expects.

    Returns shape (12,), float32.
    """
    raw = np.array(
        [
            sensors["speed"],
            sensors["heading_error"],
            sensors["checkpoint_distance"],
            *sensors["rays"],
            sensors["ground_friction"],
        ],
        dtype=np.float32,
    )
    return normalize_states(raw[None, :])[0]


def clip_action(a: np.ndarray) -> tuple[float, float]:
    """Clamp the network's (throttle, steering) output to the physical [-1, 1]
    range the controller accepts. tanh outputs are already in range, but this
    keeps you safe if you ever swap the output activation.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    throttle = float(np.clip(a[0], -1.0, 1.0))
    steering = float(np.clip(a[1], -1.0, 1.0))
    return throttle, steering

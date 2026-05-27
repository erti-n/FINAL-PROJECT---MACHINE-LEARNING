"""Steering-only policy wrapper with stuck recovery and checkpoint homing.

This is adapted from the "v14" idea: keep throttle mostly constant, use the
neural network for steering, and add explicit recovery logic when the bot is
wedged or orbiting near a checkpoint.

It is compatible with the current project weights. If the loaded network
outputs two values, the second value is used as steering and throttle is
ignored. If it outputs one value, that value is used as steering.
"""
from __future__ import annotations

import numpy as np

from drive2win import nn
from drive2win.normalize import sensors_to_input


THROTTLE = 0.90
STEER_GAIN = 1.40
STUCK_THRESHOLD = 15
REVERSE_FRAMES = 40
STUCK_SPEED = 0.30
RAY_WEDGE = 4.0
PURE_STUCK_THR = 50
PURE_STUCK_SPEED = 0.15

CP_HOMING_DIST = 50.0
CP_HOMING_MAX = 1.0
CP_HOMING_GAIN = 5.0
CP_BRAKE_DIST = 10.0
CP_MIN_THROTTLE = 0.55
CP_GATE_DIST = 5.0
CP_ORBIT_DIST = 8.0
CP_ORBIT_FRAMES = 25


def make_policy(weights_path: str):
    w = nn.load(weights_path)

    stuck_count = 0
    reverse_count = 0
    prev_cp_dist = 100.0
    orbit_frames = 0

    def policy(state: dict) -> tuple[float, float]:
        nonlocal stuck_count, reverse_count, prev_cp_dist, orbit_frames

        sensors = state["sensors"]
        speed = float(sensors.get("speed", 1.0))
        rays = list(sensors.get("rays", [50.0] * 8))
        if len(rays) < 8:
            rays = (rays + [50.0] * 8)[:8]

        front = float(rays[0])
        # Use the same side convention as the working v14 idea.
        left = float(rays[6])
        right = float(rays[2])

        wedged = (
            speed < STUCK_SPEED
            and front < RAY_WEDGE
            and (left < RAY_WEDGE or right < RAY_WEDGE)
        )
        pure_stuck = speed < PURE_STUCK_SPEED

        if wedged or pure_stuck:
            stuck_count += 1
        else:
            stuck_count = 0

        if stuck_count >= (STUCK_THRESHOLD if wedged else PURE_STUCK_THR):
            reverse_count = REVERSE_FRAMES
            stuck_count = 0

        cp_dist = float(sensors.get("checkpoint_distance", 100.0))
        if cp_dist < CP_ORBIT_DIST:
            orbit_frames += 1
        else:
            orbit_frames = 0
        if orbit_frames >= CP_ORBIT_FRAMES and reverse_count == 0:
            reverse_count = REVERSE_FRAMES * 2
            orbit_frames = 0

        if reverse_count > 0:
            reverse_count -= 1
            if reverse_count > 20:
                steer = 0.0
            else:
                steer = 0.9 if left > right else -0.9

            if front > 8.0 and reverse_count < 30:
                reverse_count = 0

            return -0.80, float(np.clip(steer, -1.0, 1.0))

        raw = np.asarray(nn.forward(sensors_to_input(sensors), w), dtype=np.float32).reshape(-1)
        steer_raw = raw[-1]
        steer = float(np.clip(steer_raw * STEER_GAIN, -1.0, 1.0))

        heading_err = float(sensors.get("heading_error", 0.0))
        old_cp_dist = prev_cp_dist
        approaching = cp_dist < old_cp_dist
        prev_cp_dist = cp_dist
        throttle = THROTTLE

        if cp_dist < CP_GATE_DIST:
            heading_norm = float(np.clip(heading_err / np.pi, -1.0, 1.0))
            steer = float(np.clip(-heading_norm * CP_HOMING_GAIN, -1.0, 1.0))
            throttle = CP_MIN_THROTTLE

        elif cp_dist < CP_ORBIT_DIST and not approaching:
            heading_norm = float(np.clip(heading_err / np.pi, -1.0, 1.0))
            steer = float(np.clip(-heading_norm * CP_HOMING_GAIN, -1.0, 1.0))
            throttle = CP_MIN_THROTTLE

        elif cp_dist < CP_HOMING_DIST:
            heading_norm = float(np.clip(heading_err / np.pi, -1.0, 1.0))
            homing_steer = float(np.clip(-heading_norm * CP_HOMING_GAIN, -1.0, 1.0))
            blend = ((CP_HOMING_DIST - cp_dist) / CP_HOMING_DIST) * CP_HOMING_MAX
            steer = steer * (1.0 - blend) + homing_steer * blend
            if cp_dist < CP_BRAKE_DIST:
                brake_t = (cp_dist - CP_GATE_DIST) / (CP_BRAKE_DIST - CP_GATE_DIST)
                brake_t = float(np.clip(brake_t, 0.0, 1.0))
                throttle = CP_MIN_THROTTLE + (THROTTLE - CP_MIN_THROTTLE) * brake_t

        return float(np.clip(throttle, -1.0, 1.0)), float(np.clip(steer, -1.0, 1.0))

    return policy

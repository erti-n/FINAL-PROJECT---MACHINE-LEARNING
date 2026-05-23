"""Hybrid navigation policy.

This keeps the learned MLP as the main driver, then adds small safety
corrections that are easier to express in code than to learn from a small
behavior-cloning dataset.
"""
from __future__ import annotations

import numpy as np

from . import nn as nn_mod
from .normalize import clip_action, sensors_to_input


def make_policy(weights_path: str):
    w = nn_mod.load(weights_path)

    def policy(state):
        sensors = state["sensors"]
        throttle, steering = clip_action(nn_mod.forward(sensors_to_input(sensors), w))

        speed = float(sensors.get("speed", 0.0))
        heading = float(sensors.get("heading_error", 0.0))
        distance = float(sensors.get("checkpoint_distance", 0.0))
        rays = list(sensors.get("rays", [50.0] * 8))
        if len(rays) < 8:
            rays = (rays + [50.0] * 8)[:8]

        front = float(rays[0])
        front_left = float(rays[1])
        front_right = float(rays[7])

        # The heading plot in the starter project expects a downward slope:
        # positive heading error usually needs negative steering correction.
        heading_correction = -0.35 * np.clip(heading / np.pi, -1.0, 1.0)
        steering = 0.70 * steering + heading_correction

        # If the learned model hesitates at the start, keep it moving unless
        # there is an obstacle directly ahead.
        if speed < 2.0 and distance > 6.0 and front > 8.0:
            throttle = max(throttle, 0.55)

        # Slow down for large heading errors so the bot turns instead of
        # sliding wide past the next checkpoint.
        if abs(heading) > 1.0:
            throttle = min(throttle, 0.45)

        # Reactive obstacle avoidance: steer toward the side with more space.
        if front < 8.0 or min(front_left, front_right) < 5.0:
            steer_right = 1.0 if front_right > front_left else -1.0
            steering = 0.55 * steering + 0.45 * steer_right
            throttle = min(throttle, 0.35)

        return clip_action(np.array([throttle, steering], dtype=np.float32))

    return policy

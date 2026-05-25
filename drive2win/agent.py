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
        nn_throttle, steering = clip_action(nn_mod.forward(sensors_to_input(sensors), w))

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

        abs_heading = abs(heading)

        # Human throttle labels are noisy when collected from key taps. Use the
        # network mainly for steering, then choose throttle from route geometry.
        if front > 15.0 and abs_heading < 0.35:
            throttle = 1.0
        elif front > 10.0 and abs_heading < 0.80:
            throttle = 0.85
        elif front > 7.0 and abs_heading < 1.30:
            throttle = 0.60
        else:
            throttle = 0.35

        # Keep moving from a standstill unless something is directly ahead.
        if speed < 2.0 and distance > 6.0 and front > 6.0:
            throttle = max(throttle, 0.75)

        # Let confident learned acceleration through, but never let a weak
        # learned throttle slow the bot on clear track.
        throttle = max(throttle, 0.25 * nn_throttle)

        # Reactive obstacle avoidance: steer toward the side with more space.
        if front < 8.0 or min(front_left, front_right) < 5.0:
            steer_right = 1.0 if front_right > front_left else -1.0
            steering = 0.55 * steering + 0.45 * steer_right
            throttle = min(throttle, 0.40)

        return clip_action(np.array([throttle, steering], dtype=np.float32))

    return policy

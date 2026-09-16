import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

import gym_openarmv2  # noqa: F401


@pytest.mark.parametrize(
    "env_task, obs_type",
    [
        ("OpenArmPickCube-v0", "pixels"),
        ("OpenArmPickCube-v0", "pixels_agent_pos"),
    ],
)
def test_openarm(env_task, obs_type):
    env = gym.make(f"gym_openarmv2/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped)

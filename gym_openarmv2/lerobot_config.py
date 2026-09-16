from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_IMAGES, OBS_STATE

from gym_openarmv2.constants import (
    CAMERA_RESOLUTIONS,
    CAMERAS,
    DEFAULT_CAMERAS,
    FPS,
    STATE_DIM,
)


@EnvConfig.register_subclass("openarmv2")
@dataclass
class OpenArmEnv(EnvConfig):
    task: str | None = "OpenArmPickCube-v0"
    fps: int = FPS
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    cameras: list[str] = field(default_factory=lambda: list(DEFAULT_CAMERAS))
    observation_height: int | None = None
    observation_width: int | None = None
    render_mode: str = "rgb_array"
    visualization_height: int = 600
    visualization_width: int = 960
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(STATE_DIM,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
        }
    )

    def __post_init__(self) -> None:
        if self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))

        for camera in self.cameras:
            # pixels/<camera> mirrors the nested observation dict.
            key = f"pixels/{camera}" if self.obs_type == "pixels_agent_pos" else camera
            self.features[key] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(*self._resolution(camera), 3)
            )
            self.features_map[key] = f"{OBS_IMAGES}.{camera}"

        if self.obs_type == "pixels" and len(self.cameras) == 1:
            self.features_map[self.cameras[0]] = OBS_IMAGE

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "cameras": self.cameras,
            "observation_height": self.observation_height,
            "observation_width": self.observation_width,
            "render_mode": self.render_mode,
            "visualization_height": self.visualization_height,
            "visualization_width": self.visualization_width,
            "max_episode_steps": self.episode_length,
        }

    def _resolution(self, camera: str) -> tuple[int, int]:
        if self.observation_height and self.observation_width:
            return self.observation_height, self.observation_width
        if camera not in CAMERA_RESOLUTIONS:
            raise ValueError(f"unknown camera {camera!r}, expected from {CAMERAS}")
        return CAMERA_RESOLUTIONS[camera]

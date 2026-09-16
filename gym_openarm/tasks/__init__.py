from __future__ import annotations

from gym_openarm.tasks.base import Task
from gym_openarm.tasks.pick_cube import PickCubeTask

TASKS: dict[str, type[Task]] = {
    "pick_cube": PickCubeTask,
}


def make_task(name: str, **kwargs) -> Task:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}, expected one of {sorted(TASKS)}")
    return TASKS[name](**kwargs)


__all__ = ["TASKS", "PickCubeTask", "Task", "make_task"]

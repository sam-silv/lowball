"""Load task instances from YAML files."""

from pathlib import Path

import yaml

from lowball.tasks.schema import TaskInstance

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "tasks"


def load_task(task_id: str, data_dir: Path = DATA_DIR) -> TaskInstance:
    """Load a single task instance by ID (e.g., 'easy/civic_2019')."""
    task_path = data_dir / f"{task_id}.yaml"
    if not task_path.exists():
        raise FileNotFoundError(f"Task not found: {task_path}")

    with open(task_path) as f:
        raw = yaml.safe_load(f)

    raw["task_id"] = task_id
    return TaskInstance(**raw)


def load_suite(suite: str, data_dir: Path = DATA_DIR) -> list[TaskInstance]:
    """Load all tasks in a difficulty suite (e.g., 'easy')."""
    suite_dir = data_dir / suite
    if not suite_dir.is_dir():
        raise FileNotFoundError(f"Suite directory not found: {suite_dir}")

    tasks: list[TaskInstance] = []
    for task_file in sorted(suite_dir.glob("*.yaml")):
        task_id = f"{suite}/{task_file.stem}"
        tasks.append(load_task(task_id, data_dir))

    return tasks

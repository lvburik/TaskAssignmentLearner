"""Build a current-state HDF5 dataset for the XGBOOST test model.

Each row represents one agent/task pair at one timestep. The features contain
only the current state; no assignments from earlier timesteps are included.
The target ``y`` is one when the task is assigned to the candidate agent.

Example:

    .venv/bin/python XGBoost_dataset_builder.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _position(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"Expected a two-dimensional position, got {value!r}")
    return float(value[0]), float(value[1])


def _task_position(value: Any) -> tuple[float, float]:
    """Read both TASim task formats: [x, y] and [[x, y], metadata]."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Expected a task position, got {value!r}")
    if isinstance(value[0], (list, tuple)):
        return _position(value[0])
    return _position(value)


def _read_states(path: Path) -> tuple[list[dict[str, Any]], int, int]:
    states: list[dict[str, Any]] = []
    world_width = world_height = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from error

            if "simulation_paramaters" in record:
                continue
            if "world_info" in record:
                map_matrix = record["world_info"].get("map_matrix")
                if isinstance(map_matrix, list) and map_matrix:
                    world_height = len(map_matrix)
                    world_width = (
                        len(map_matrix[0]) if isinstance(map_matrix[0], list) else 0
                    )
                continue
            if len(record) != 1:
                raise ValueError(
                    f"Expected one timestep record in {path} line {line_number}"
                )
            timestep, state = next(iter(record.items()))
            if not isinstance(state, dict):
                raise ValueError(f"Expected a state object in {path} line {line_number}")
            state["_timestep"] = int(timestep)
            states.append(state)

    states.sort(key=lambda state: state["_timestep"])
    if not states:
        raise ValueError(f"{path} does not contain any timestep records")
    return states, world_width, world_height


def _rows(path: Path, max_timesteps: int | None) -> Iterable[dict[str, float | int]]:
    states, world_width, world_height = _read_states(path)
    if max_timesteps is not None:
        if max_timesteps < 1:
            raise ValueError("max_timesteps must be at least 1")
        states = states[:max_timesteps]

    for state in states:
        agents = state.get("agents", {})
        tasks = state.get("tasks", {})
        if not isinstance(agents, dict) or not isinstance(tasks, dict):
            raise ValueError(f"{path} timestep {state['_timestep']} has invalid agents/tasks")

        positions = {
            str(agent_id): _position(agent["pos"]) for agent_id, agent in agents.items()
        }
        team_size = len(positions)
        for agent_id, agent in agents.items():
            agent_x, agent_y = positions[str(agent_id)]
            assigned_tasks = {str(task_id) for task_id in agent.get("tasks", [])}

            for task_id, task in tasks.items():
                task_x, task_y = _task_position(task)
                distance = abs(agent_x - task_x) + abs(agent_y - task_y)
                teammate_distances = [
                    abs(other_x - task_x) + abs(other_y - task_y)
                    for other_id, (other_x, other_y) in positions.items()
                    if other_id != str(agent_id)
                ]
                closer_agents_count = sum(
                    teammate_distance < distance
                    for teammate_distance in teammate_distances
                )
                yield {
                    "task_x": task_x,
                    "task_y": task_y,
                    "agent_x": agent_x,
                    "agent_y": agent_y,
                    "manhattan_distance": distance,
                    "team_size": team_size,
                    "closer_agents_count": closer_agents_count,
                    "closest_teammate_manhattan_distance": min(teammate_distances)
                    if teammate_distances
                    else distance,
                    "world_width": world_width,
                    "world_height": world_height,
                    "y": int(str(task_id) in assigned_tasks),
                }


def build_dataset(
    input_dir: Path,
    output: Path,
    max_timesteps: int | None = None,
) -> tuple[int, int]:
    """Write all current-state agent/task rows to an HDF5 file."""
    paths = sorted(input_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No .json files found in {input_dir}")

    rows = [row for path in paths for row in _rows(path, max_timesteps)]
    if not rows:
        raise ValueError("Input simulations contain no agent/task pairs")

    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_hdf(
        output,
        key="data",
        mode="w",
        format="fixed",
        complevel=5,
        complib="blosc",
    )
    return len(frame), int(frame["y"].sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("JSON_Data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("DataSets/XGBoosttest.h5"),
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Process only the first N timesteps in each simulation.",
    )
    args = parser.parse_args()
    row_count, positive_count = build_dataset(
        args.input_dir, args.output, args.max_timesteps
    )
    print(f"Saved {row_count:,} rows ({positive_count:,} positive) to {args.output}")


if __name__ == "__main__":
    main()

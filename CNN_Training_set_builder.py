"""Build a CNN-ready HDF5 dataset from one or more TASim JSONL files.

The HDF5 file contains:

``images``
    ``uint8`` samples with shape ``(N, height, width, 3)``. The channels are
    the map, all tasks and all agents. The candidate task and agent are
    highlighted in channels 1 and 2 respectively.
``labels``
    ``uint8`` values with shape ``(N,)``; 1 means that the task is assigned to
    the agent in the current timestep.
``source_file`` and ``timestep``
    Metadata for tracing samples back to the simulation.

Example:

    python CNN_Training_set_builder.py --input-dir JSON_Data \
        --output DataSets/cnn_training.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


def _read_simulation(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Read metadata and timestep states from one TASim JSONL file."""
    parameters: dict[str, Any] = {}
    world_info: dict[str, Any] = {}
    states: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from error

            if "simulation_paramaters" in record:
                parameters = record["simulation_paramaters"]
            elif "world_info" in record:
                world_info = record["world_info"]
            else:
                if len(record) != 1:
                    raise ValueError(
                        f"Expected one timestep record in {path} line {line_number}"
                    )
                timestep, state = next(iter(record.items()))
                state["_timestep"] = int(timestep)
                states.append(state)

    states.sort(key=lambda state: state["_timestep"])
    if not states:
        raise ValueError(f"{path} does not contain any timesteps")
    if "map_matrix" not in world_info:
        raise ValueError(f"{path} is missing world_info.map_matrix")
    return parameters, world_info, states


def _position(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"Expected a two-dimensional position, got {value!r}")
    return int(value[0]), int(value[1])


def _world_state_matrix(map_matrix: Any, world_state: dict[str, Any]) -> np.ndarray:
    """Create an ``(height, width, 3)`` image for one world state."""
    map_channel = np.asarray(map_matrix, dtype=np.float32)
    if map_channel.ndim != 2:
        raise ValueError("world_info.map_matrix must be a two-dimensional matrix")

    height, width = map_channel.shape
    task_channel = np.zeros((height, width), dtype=np.float32)
    for task_location in world_state.get("tasks", {}).values():
        x, y = _position(task_location)
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Task position {(x, y)} is outside the map")
        task_channel[y, x] = 0.5

    agent_channel = np.zeros((height, width), dtype=np.float32)
    for agent in world_state.get("agents", {}).values():
        x, y = _position(agent["pos"])
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Agent position {(x, y)} is outside the map")
        agent_channel[y, x] = 0.5

    return np.clip(
        np.stack((map_channel, task_channel, agent_channel), axis=-1) * 255,
        0,
        255,
    ).astype(np.uint8)


def _training_datapoints(
    path: Path,
) -> Iterable[tuple[np.ndarray, int, str, int]]:
    """Yield CNN samples from every agent/task pair in a simulation."""
    _, world_info, states = _read_simulation(path)
    for state in states:
        state_image = _world_state_matrix(world_info["map_matrix"], state)
        for agent in state.get("agents", {}).values():
            agent_x, agent_y = _position(agent["pos"])
            assigned_tasks = {str(task_id) for task_id in agent.get("tasks", [])}
            for task_id, task_location in state.get("tasks", {}).items():
                task_x, task_y = _position(task_location)
                image = state_image.copy()
                image[task_y, task_x, 1] = 255
                image[agent_y, agent_x, 2] = 255
                yield image, int(task_id in assigned_tasks), path.name, state["_timestep"]


def build_dataset(input_dir: Path, output: Path) -> int:
    """Convert all JSON files in ``input_dir`` into one HDF5 file."""
    paths = sorted(input_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No .json files found in {input_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    sample_count = 0
    with h5py.File(output, "w") as dataset:
        images: h5py.Dataset | None = None
        labels = dataset.create_dataset(
            "labels", shape=(0,), maxshape=(None,), dtype=np.uint8, chunks=True
        )
        timesteps = dataset.create_dataset(
            "timestep", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True
        )
        source_files = dataset.create_dataset(
            "source_file",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=True,
        )

        for path in paths:
            for image, label, source_file, timestep in _training_datapoints(path):
                if images is None:
                    images = dataset.create_dataset(
                        "images",
                        shape=(0,) + image.shape,
                        maxshape=(None,) + image.shape,
                        dtype=np.uint8,
                        chunks=(1,) + image.shape,
                        compression="gzip",
                    )
                elif image.shape != images.shape[1:]:
                    raise ValueError(
                        f"Map shape for {path} differs from the first input file"
                    )

                new_size = sample_count + 1
                images.resize((new_size,) + images.shape[1:])
                labels.resize((new_size,))
                timesteps.resize((new_size,))
                source_files.resize((new_size,))
                images[sample_count] = image
                labels[sample_count] = label
                timesteps[sample_count] = timestep
                source_files[sample_count] = source_file
                sample_count = new_size

        if images is None:
            raise ValueError("Input simulations contain no agent/task pairs")
        dataset.attrs["image_format"] = "NHWC"
        dataset.attrs["description"] = "TASim CNN assignment training samples"

    return sample_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("JSON_Data"))
    parser.add_argument(
        "--output", type=Path, default=Path("DataSets/cnn_training.h5")
    )
    args = parser.parse_args()
    count = build_dataset(args.input_dir, args.output)
    print(f"Saved {count:,} samples to {args.output}")


if __name__ == "__main__":
    main()

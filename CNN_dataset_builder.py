"""Build a CNN-ready HDF5 dataset from TASim JSONL simulations.

Each sample represents one candidate ``(agent, task)`` pair at one timestep.
The image has three channels in NHWC order:

* channel 0: the map matrix;
* channel 1: all task locations, with the candidate task highlighted;
* channel 2: all agent locations, with the candidate agent highlighted.

Empty task and agent pixels use intensity 128, regular locations use 128, and
the candidate locations use 255. The label is 1 when the candidate task is
assigned to the candidate agent in that timestep.

Example:

    python CNN_dataset_builder.py --input-dir JSON_Data \
        --output DataSets/cnn_training.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np


def _position(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"Expected a two-dimensional position, got {value!r}")
    return int(value[0]), int(value[1])


def _task_position(value: Any) -> tuple[int, int]:
    """Read both TASim task formats: ``[x, y]`` and ``[[x, y], metadata]``."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Expected a task position, got {value!r}")
    return _position(value[0]) if isinstance(value[0], (list, tuple)) else _position(value)


def _read_simulation(
    path: Path,
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    map_matrix: list[list[Any]] | None = None
    states: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from error

            if "world_info" in record:
                candidate = record["world_info"].get("map_matrix")
                if not isinstance(candidate, list) or not candidate:
                    raise ValueError(f"{path} has an invalid world_info.map_matrix")
                map_matrix = candidate
                continue
            if "simulation_paramaters" in record:
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

    if map_matrix is None:
        raise ValueError(f"{path} is missing world_info.map_matrix")
    states.sort(key=lambda state: state["_timestep"])
    if not states:
        raise ValueError(f"{path} does not contain any timesteps")
    return map_matrix, states


def _base_image(map_matrix: list[list[Any]], state: dict[str, Any]) -> np.ndarray:
    map_channel = np.asarray(map_matrix, dtype=np.float32)
    if map_channel.ndim != 2 or map_channel.size == 0:
        raise ValueError("world_info.map_matrix must be a non-empty 2D matrix")
    height, width = map_channel.shape
    task_channel = np.full((height, width), 128, dtype=np.float32)
    agent_channel = np.full((height, width), 128, dtype=np.float32)

    for task in state.get("tasks", {}).values():
        x, y = _task_position(task)
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Task position {(x, y)} is outside the map")
        task_channel[y, x] = 192
    for agent in state.get("agents", {}).values():
        x, y = _position(agent["pos"])
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Agent position {(x, y)} is outside the map")
        agent_channel[y, x] = 192

    map_channel = np.clip(map_channel * 255, 0, 255)
    return np.stack((map_channel, task_channel, agent_channel), axis=-1).astype(np.uint8)


def _samples(
    path: Path, max_timesteps: int | None
) -> Iterator[tuple[np.ndarray, int, str, str, int]]:
    map_matrix, states = _read_simulation(path)
    if max_timesteps is not None:
        if max_timesteps < 1:
            raise ValueError("max_timesteps must be at least 1")
        states = states[:max_timesteps]

    for state in states:
        agents = state.get("agents", {})
        tasks = state.get("tasks", {})
        if not isinstance(agents, dict) or not isinstance(tasks, dict):
            raise ValueError(f"{path} timestep {state['_timestep']} has invalid agents/tasks")
        state_image = _base_image(map_matrix, state)
        for agent_id, agent in agents.items():
            agent_x, agent_y = _position(agent["pos"])
            assigned_tasks = {str(task_id) for task_id in agent.get("tasks", [])}
            for task_id, task in tasks.items():
                task_x, task_y = _task_position(task)
                image = state_image.copy()
                image[task_y, task_x, 1] = 255
                image[agent_y, agent_x, 2] = 255
                yield (
                    image,
                    int(str(task_id) in assigned_tasks),
                    str(agent_id),
                    str(task_id),
                    int(state["_timestep"]),
                )


def build_dataset(
    input_dir: Path,
    output: Path,
    max_timesteps: int | None = None,
    chunk_size: int = 1024,
) -> tuple[int, int]:
    """Write all agent/task samples to an HDF5 file and return counts."""
    paths = sorted(input_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No .json files found in {input_dir}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    output.parent.mkdir(parents=True, exist_ok=True)
    sample_count = positive_count = 0
    with h5py.File(output, "w") as dataset:
        images: h5py.Dataset | None = None
        labels = dataset.create_dataset("labels", shape=(0,), maxshape=(None,), dtype=np.uint8)
        agent_ids = dataset.create_dataset("agent_id", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
        task_ids = dataset.create_dataset("task_id", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
        timesteps = dataset.create_dataset("timestep", shape=(0,), maxshape=(None,), dtype=np.int64)

        batch: list[tuple[np.ndarray, int, str, str, int]] = []

        def write_batch() -> None:
            nonlocal images, sample_count, positive_count
            if not batch:
                return
            batch_images = np.stack([item[0] for item in batch])
            if images is None:
                images = dataset.create_dataset(
                    "images",
                    shape=(0,) + batch_images.shape[1:],
                    maxshape=(None,) + batch_images.shape[1:],
                    dtype=np.uint8,
                    chunks=(1,) + batch_images.shape[1:],
                    compression="gzip",
                )
            elif images.shape[1:] != batch_images.shape[1:]:
                raise ValueError("Input simulations must all use the same map shape")
            end = sample_count + len(batch)
            images.resize((end,) + images.shape[1:])
            labels.resize((end,))
            agent_ids.resize((end,))
            task_ids.resize((end,))
            timesteps.resize((end,))
            images[sample_count:end] = batch_images
            labels[sample_count:end] = [item[1] for item in batch]
            agent_ids[sample_count:end] = [item[2] for item in batch]
            task_ids[sample_count:end] = [item[3] for item in batch]
            timesteps[sample_count:end] = [item[4] for item in batch]
            positive_count += sum(item[1] for item in batch)
            sample_count = end
            batch.clear()

        for path in paths:
            for sample in _samples(path, max_timesteps):
                batch.append(sample)
                if len(batch) >= chunk_size:
                    write_batch()
        write_batch()

        if images is None:
            raise ValueError("Input simulations contain no agent/task pairs")
        dataset.attrs["image_format"] = "NHWC"
        dataset.attrs["description"] = "TASim CNN assignment training samples"

    return sample_count, positive_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("JSON_Data"))
    parser.add_argument("--output", type=Path, default=Path("DataSets/cnn_training.h5"))
    parser.add_argument("--max-timesteps", type=int, default=None)
    args = parser.parse_args()
    sample_count, positive_count = build_dataset(
        args.input_dir, args.output, args.max_timesteps
    )
    print(
        f"Saved {sample_count:,} samples ({positive_count:,} positive, "
        f"{sample_count - positive_count:,} negative) to {args.output}"
    )


if __name__ == "__main__":
    main()

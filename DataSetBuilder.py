"""Build supervised examples from TASim line-separated JSON simulations.

Each output row is one possible ``(agent, task)`` assignment at timestep ``t``.
The target is 1 when the task is newly assigned to that agent at timestep
``t + 1``. Variable-length context is kept as JSON arrays so no information is
lost before a model-specific preprocessing step.

Example:
	.venv/bin/python DataSetBuilder.py --input-dir JSON_Data \
		--output dataset.jsonl.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable, TextIO


def _open_text(path: Path, mode: str) -> TextIO:
	"""Open plain or gzip-compressed UTF-8 text based on the filename."""
	if path.suffix == ".gz":
		return gzip.open(path, mode + "t", encoding="utf-8")
	return path.open(mode, encoding="utf-8")


def _read_simulation(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
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
				timestep, state = next(iter(record.items()))
				state["_timestep"] = int(timestep)
				states.append(state)

	states.sort(key=lambda state: state["_timestep"])
	if len(states) < 2:
		raise ValueError(f"{path} must contain at least two timesteps")
	return parameters, world_info, states


def _position(value: Any) -> list[float]:
	"""Convert a simulator position to a JSON-friendly two-number list."""
	if not isinstance(value, (list, tuple)) or len(value) < 2:
		raise ValueError(f"Expected a two-dimensional position, got {value!r}")
	return [float(value[0]), float(value[1])]


def _task_position(task: Any) -> list[float]:
	"""Extract the position from the TASim task format: [position, metadata]."""
	if not isinstance(task, (list, tuple)) or not task:
		raise ValueError(f"Expected a task [position, metadata] pair, got {task!r}")
	return _position(task[0])


def _json(value: Any) -> str:
	return json.dumps(value, separators=(",", ":"))


def _rows(
	path: Path,
	negative_ratio: float | None,
	max_timesteps: int | None,
) -> Iterable[dict[str, Any]]:
	parameters, world_info, states = _read_simulation(path)
	if max_timesteps is not None:
		states = states[: max_timesteps + 1]

	for index, (state, next_state) in enumerate(zip(states, states[1:])):
		agents = state["agents"]
		next_agents = next_state["agents"]
		tasks = state["tasks"]
		previous_tasks = states[index - 1]["tasks"] if index else {}
		previous_agent_tasks = states[index - 1]["agents"] if index else {}

		task_ids = sorted(tasks, key=lambda task_id: int(task_id))
		agent_ids = sorted(agents, key=lambda agent_id: int(agent_id))
		next_assignments = {
			(str(agent_id), int(task_id))
			for agent_id, agent in next_agents.items()
			for task_id in agent.get("tasks", [])
		}

		positives = []
		negatives = []
		for agent_id in agent_ids:
			current_task_ids = {int(task_id) for task_id in agents[agent_id].get("tasks", [])}
			for task_id in task_ids:
				task_number = int(task_id)
				label = int((str(agent_id), task_number) in next_assignments and task_number not in current_task_ids)
				row = {
					"source_file": path.name,
					"timestep": state["_timestep"],
					"agent_id": int(agent_id),
					"task_id": task_number,
					"label_assign_next": label,
					"agent_x": _position(agents[agent_id]["pos"])[0],
					"agent_y": _position(agents[agent_id]["pos"])[1],
					"task_x": _task_position(tasks[task_id])[0],
					"task_y": _task_position(tasks[task_id])[1],
					"is_currently_assigned": int(task_number in current_task_ids),
					"previous_agent_task_ids": _json(previous_agent_tasks.get(agent_id, {}).get("tasks", [])),
					"previous_agent_task_locations": _json([
						_task_position(previous_tasks[str(task)])
						for task in previous_agent_tasks.get(agent_id, {}).get("tasks", [])
						if str(task) in previous_tasks
					]),
					"all_task_locations": _json([_task_position(tasks[task]) for task in task_ids]),
					"other_agent_locations": _json([
						_position(agents[other_id]["pos"])
						for other_id in agent_ids
						if other_id != agent_id
					]),
					"number_of_agents": parameters.get("number_of_agents", len(agents)),
					"number_of_tasks": parameters.get("number_of_tasks", len(tasks)),
					"map_width": world_info.get("map_width"),
					"map_height": world_info.get("map_height"),
				}
				(positives if label else negatives).append(row)

		if negative_ratio is None:
			yield from positives
			yield from negatives
		else:
			yield from positives
			limit = int(len(positives) * negative_ratio)
			yield from negatives[:limit]


def build_dataset(
	input_dir: Path,
	output: Path,
	negative_ratio: float | None = None,
	max_timesteps: int | None = None,
) -> tuple[int, int]:
	"""Build a JSONL/CSV dataset and return ``(rows, positive_rows)``."""
	files = sorted(input_dir.glob("*.json"))
	if not files:
		raise FileNotFoundError(f"No .json files found in {input_dir}")
	if negative_ratio is not None and negative_ratio < 0:
		raise ValueError("negative_ratio must be non-negative")

	output.parent.mkdir(parents=True, exist_ok=True)
	is_csv = output.suffix == ".csv"
	fieldnames: list[str] | None = None
	rows = positives = 0
	with _open_text(output, "w") as destination:
		writer = None
		for path in files:
			for row in _rows(path, negative_ratio, max_timesteps):
				if fieldnames is None:
					fieldnames = list(row)
					if is_csv:
						writer = csv.DictWriter(destination, fieldnames=fieldnames)
						writer.writeheader()
				if is_csv:
					assert writer is not None
					writer.writerow(row)
				else:
					destination.write(json.dumps(row) + "\n")
				rows += 1
				positives += row["label_assign_next"]

	return rows, positives


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input-dir", type=Path, default=Path("JSON_Data"))
	parser.add_argument("--output", type=Path, default=Path("dataset.jsonl.gz"))
	parser.add_argument(
		"--negative-ratio",
		type=float,
		default=None,
		help="Keep at most this many negative rows per positive row; default keeps all rows.",
	)
	parser.add_argument(
		"--max-timesteps",
		type=int,
		default=None,
		help="Process only the first N transitions per simulation, useful for a quick test.",
	)
	args = parser.parse_args()
	row_count, positive_count = build_dataset(
		args.input_dir, args.output, args.negative_ratio, args.max_timesteps
	)
	print(f"Wrote {row_count:,} rows ({positive_count:,} positive) to {args.output}")


if __name__ == "__main__":
	main()


import json
from pathlib import Path

import pandas as pd


def create_dataset(data_folder, dataset_name):
    data_folder = Path(data_folder)
    row_list = []
    for file_path in sorted(data_folder.glob("*.json")):
        with file_path.open() as rf:
            for line_number, line in enumerate(rf):
                content = json.loads(line)

                # Ignore the first few timesteps.
                if line_number - 2 <= 5:
                    continue

                timestep = line_number - 2
                print(f"processing {file_path.name}, timestep {timestep}/500")
                content = content[str(timestep)]
                agent_locations = [
                    agent["pos"] for agent in content["agents"].values()
                ]

                for agent in content["agents"].values():
                    agent_location = agent["pos"]
                    team_locations = [
                        location
                        for location in agent_locations
                        if location != agent_location
                    ]
                    assigned_tasks = {str(task_id) for task_id in agent["tasks"]}

                    for task_id, task in content["tasks"].items():
                        row = {
                            "task_x": task[0],
                            "task_y": task[1],
                            "agent_x": agent_location[0],
                            "agent_y": agent_location[1],
                            "team_count": len(team_locations),
                            "y": str(task_id) in assigned_tasks,
                        }
                        for teammate_index, location in enumerate(team_locations):
                            row[f"teammate_{teammate_index}_x"] = location[0]
                            row[f"teammate_{teammate_index}_y"] = location[1]
                        row_list.append(row)

    df = pd.DataFrame(row_list).fillna(0)
    output_path = Path("DataSets") / f"{dataset_name}.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(
        output_path,
        key="data",
        mode="w",
        format="fixed",
        complevel=5,
        complib="blosc",
    )
    print(f"Saved {len(df):,} rows to {output_path}")


create_dataset("Data/trainset", "train1")
create_dataset("Data/testset", "test1")



                
                        
                        
                




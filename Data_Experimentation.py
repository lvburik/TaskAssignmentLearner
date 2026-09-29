
import os



import json
import pandas as pd
def create_dataset(DATA_FOLDER, dataset_name):
    files = os.listdir(DATA_FOLDER)
    row_list = []
    for file in files:
        with open(DATA_FOLDER + file) as rf:
                
                for ii, line in enumerate(rf) :
                    content = json.loads(line)

                    #ignore first few timesteps
                    if ii-2>5:
                        print(f"processing timestep {ii-2}/500")
                        content = content[str(ii-2)]
                        agent_locations = [content['agents'][agent_id]['pos'] for agent_id in content['agents'].keys()]
                        


                        for agent_id in content['agents'].keys():
                            aloc = content['agents'][agent_id]['pos']
                            alocs = [agent_location for agent_location in agent_locations if agent_location != aloc]

                            assigned_tasks = {str(task_id) for task_id in content['agents'][agent_id]['tasks']}       
                            for task_id in content['tasks'].keys():
                                row = {}
                                row['X'] = {'task_loc': content['tasks'][task_id][0], 'agent_loc': aloc, 'team_loc': alocs}
                                row['y'] = row['y'] = str(task_id) in assigned_tasks
                                row_list.append(row)
                            
    
    df = pd.DataFrame(row_list)

    df.to_csv("DataSets/"+dataset_name+".csv")

create_dataset("Data/", "set1")



                
                        
                        
                



